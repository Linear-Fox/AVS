#ifndef AUDIO_VISUAL_SYNTHESIS_H
#define AUDIO_VISUAL_SYNTHESIS_H

#include <tensorflow/core/framework/tensor.h>

class AudioVisualSynthesis {
public:
    AudioVisualSynthesis(const std::string& model_path);
    ~AudioVisualSynthesis();

    bool LoadModel();
    tensorflow::Tensor ProcessInput(const tensorflow::Tensor& input_tensor);
    tensorflow::Tensor GenerateOutput(const tensorflow::Tensor& input_tensor);
private:
    std::unique_ptr<tensorflow::Session> session_;
    std::string model_path_;
};

#endif