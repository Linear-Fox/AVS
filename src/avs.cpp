#include "avs.h"

AudioVisualSynthesis::AudioVisualSynthesis(const std::string & audio_model_path,
  const std::string & video_model_path,
    const std::string & input_audio_path,
      const std::string & input_video_path,
        const std::string & output_audio_path,
          const std::string & output_video_path) {
  audio_model_ = LoadModel(audio_model_path);
  video_model_ = LoadModel(video_model_path);

  input_audio_ = LoadAudio(input_audio_path);
  input_video_ = LoadVideo(input_video_path);

  output_audio_path_ = output_audio_path;
  output_video_path_ = output_video_path;
}

AudioVisualSynthesis::~AudioVisualSynthesis() {
  delete[] input_audio_.data;
  cv::release(input_video_);
  cv::release(output_video_);
}

tensorflow::Status AudioVisualSynthesis::LoadModel(const std::string & model_path) {
  tensorflow::SessionOptions session_options;
  session_options.config.mutable_gpu_options() -> set_allow_growth(true);
  tensorflow::Status status = tensorflow::NewSession(session_options, & session_);
  if (!status.ok()) {
    return status;
  }

  tensorflow::GraphDef graph_def;
  status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, & graph_def);
  if (!status.ok()) {
    return status;
  }

  status = session_ -> Create(graph_def);
  if (!status.ok()) {
    return status;
  }

  return tensorflow::Status::OK();
}

AudioVisualSynthesis::AudioData AudioVisualSynthesis::LoadAudio(const std::string & audio_path) {
  AudioData audio_data;

  drwav wav;
  drwav_init_file( & wav, audio_path.c_str(), NULL);
  audio_data.sample_rate = wav.sampleRate;
  audio_data.num_samples = static_cast < int > (wav.totalPCMFrameCount);
  audio_data.num_channels = wav.channels;
  audio_data.data = new float[audio_data.num_samples];

  drwav_read_pcm_frames_f32( & wav, audio_data.num_samples, audio_data.data);
  drwav_uninit( & wav);

  return audio_data;
}

tensorflow::Status AudioVisualSynthesis::Run() {
  tensorflow::Tensor audio_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({
    1,
    input_audio_.num_samples,
    1,
    input_audio_.num_channels
  }));
  std::copy(input_audio_.data, input_audio_.data + input_audio_.num_samples * input_audio_.num_channels, audio_tensor.flat < float > ().data());

  tensorflow::Tensor video_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({
    1,
    input_video_.rows,
    input_video_.cols,
    input_video_.channels()
  }));
  cv::Mat float_video;
  input_video_.convertTo(float_video, CV_32FC3);
  std::memcpy(video_tensor.flat < float > ().data(), float_video.data, input_video_.total() * input_video_.elemSize());

  std::vector < tensorflow::Tensor > audio_outputs;
  tensorflow::Status audio_status = session_ -> Run({
    {
      "audio_input",
      audio_tensor
    }
  }, {
    "audio_output"
  }, {}, & audio_outputs);
  if (!audio_status.ok()) {
    return audio_status;
  }

  std::vectortensorflow::Tensor video_outputs;
  tensorflow::Status video_status = session_ -> Run({
    {
      "video_input",
      video_tensor
    }
  }, {
    "video_output"
  }, {}, & video_outputs);
  if (!video_status.ok()) {
    return video_status;
  }

  tensorflow::Tensor audio_output_tensor = audio_outputs[0];
  tensorflow::Tensor video_output_tensor = video_outputs[0];
 
  SaveAudio(output_audio_path_, audio_output_tensor.flat < float > ().data(), audio_output_tensor.dim_size(1), audio_output_tensor.dim_size(3), input_audio_.sample_rate);
 
  cv::Mat output_video(input_video_.size(), CV_32FC3, video_output_tensor.flat < float > ().data());
  output_video.convertTo(output_video_, CV_8UC3);
  cv::imwrite(output_video_path_, output_video_);

  return tensorflow::Status::OK();
}
 
void AudioVisualSynthesis::SaveAudio(const std::string & audio_path,
    const float * audio_data, int num_samples, int num_channels, int sample_rate) {
    drwav_data_format format;
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_IEEE_FLOAT;
    format.channels = num_channels;
    format.sampleRate = sample_rate;
    format.bitsPerSample = 32;
    drwav * wav = drwav_open_file_write(audio_path.c_str(), & format);
    if (wav != NULL) {
        drwav_write_pcm_frames(wav, num_samples / num_channels, audio_data);
        drwav_close(wav);
    }
}