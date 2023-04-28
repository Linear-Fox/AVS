#ifndef AUDIO_VISUAL_SYNTHESIS_H_
#define AUDIO_VISUAL_SYNTHESIS_H_

#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <dr_wav.h>

struct AudioData {
  float* data;
  int num_samples;
  int num_channels;
  int sample_rate;
};

struct VideoData {
  cv::Mat data;
  int num_frames;
  int width;
  int height;
};

class AudioVisualSynthesis {
 public:
  AudioVisualSynthesis(const std::string& audio_model_path, const std::string& video_model_path);
  ~AudioVisualSynthesis();

  tensorflow::Status Synthesize(const std::string& input_audio_path, const std::string& input_video_path, const std::string& output_audio_path, const std::string& output_video_path);

 private:
  void SaveAudio(const std::string& audio_path, const float* audio_data, int num_samples, int num_channels, int sample_rate);

  tensorflow::Session* audio_session_;
  tensorflow::Session* video_session_;
  tensorflow::Status audio_session_status_;
  tensorflow::Status video_session_status_;
  tensorflow::GraphDef audio_graph_;
  tensorflow::GraphDef video_graph_;
  AudioData input_audio_;
  VideoData input_video_;
  cv::Mat output_video_;
};

#endif  // AUDIO_VISUAL_SYNTHESIS_H_