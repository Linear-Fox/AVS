#include "avs.h"

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <audio_model_path> <video_model_path> <input_audio_path> <input_video_path>" << std::endl;
    return 1;
  }

  const std::string audio_model_path = argv[1];
  const std::string video_model_path = argv[2];
  const std::string input_audio_path = argv[3];
  const std::string input_video_path = argv[4];
  const std::string output_audio_path = "output.wav";
  const std::string output_video_path = "output.jpg";

  AudioVisualSynthesis avs(audio_model_path, video_model_path);
  tensorflow::Status status = avs.Synthesize(input_audio_path, input_video_path, output_audio_path, output_video_path);
  if (!status.ok()) {
    std::cerr << "Error synthesizing audio and video: " << status.ToString() << std::endl;
    return 1;
  }

  std::cout << "Audio and video synthesized successfully!" << std::endl;
  return 0;
}
