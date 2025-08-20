/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/llm/runner/multimodal_runner.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <iostream>
#include <iomanip>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace ::executorch::extension::llm;

// Simple image generator for testing
Image create_test_gradient_image() {
    Image image;
    image.width = 224;
    image.height = 224; 
    image.channels = 3;
    
    image.data.resize(image.width * image.height * image.channels);
    
    // Create a gradient pattern
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < image.channels; ++c) {
                int idx = (y * image.width + x) * image.channels + c;
                if (c == 0) { // Red
                    image.data[idx] = static_cast<uint8_t>((x * 255) / image.width);
                } else if (c == 1) { // Green  
                    image.data[idx] = static_cast<uint8_t>((y * 255) / image.height);
                } else { // Blue
                    image.data[idx] = static_cast<uint8_t>(((x + y) * 128) / (image.width + image.height));
                }
            }
        }
    }
    
    return image;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.pte> <tokenizer_path>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    std::string tokenizer_path = argv[2];
    
    std::cout << "🚀 Multimodal Runner Example" << std::endl;
    std::cout << "============================" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Tokenizer: " << tokenizer_path << std::endl;
    std::cout << std::endl;
    
    try {
        // 1. Load tokenizer
        std::cout << "1. 🔧 Loading tokenizer..." << std::endl;
        auto tokenizer = load_tokenizer(tokenizer_path);
        if (!tokenizer) {
            std::cerr << "Failed to load tokenizer from: " << tokenizer_path << std::endl;
            return 1;
        }
        std::cout << "   └── Tokenizer loaded successfully ✅" << std::endl;
        std::cout << std::endl;
        
        // 2. Create multimodal runner
        std::cout << "2. 🏗️  Creating multimodal runner..." << std::endl;
        auto runner = create_multimodal_runner(model_path.c_str(), std::move(tokenizer));
        if (!runner) {
            std::cerr << "Failed to create multimodal runner" << std::endl;
            return 1;
        }
        std::cout << "   └── Multimodal runner created successfully ✅" << std::endl;
        std::cout << std::endl;
        
        // 3. Load model
        std::cout << "3. 📥 Loading model..." << std::endl;
        auto load_result = runner->load();
        if (load_result != Error::Ok) {
            std::cerr << "Failed to load model" << std::endl;
            return 1;
        }
        std::cout << "   └── Model loaded successfully ✅" << std::endl;
        std::cout << std::endl;
        
        // 4. Create multimodal inputs
        std::cout << "4. 🖼️  Creating multimodal inputs..." << std::endl;
        std::vector<MultimodalInput> inputs;
        
        // Add text input
        inputs.emplace_back(make_text_input("What do you see in this image?"));
        std::cout << "   ├── Text input created ✅" << std::endl;
        
        // Add test image
        Image test_image = create_test_gradient_image();
        inputs.emplace_back(make_image_input(std::move(test_image)));
        std::cout << "   └── Image input created (224x224 test gradient) ✅" << std::endl;
        std::cout << std::endl;
        
        // 5. Configure generation
        std::cout << "5. ⚙️  Setting generation config..." << std::endl;
        GenerationConfig config;
        config.max_new_tokens = 100;
        config.temperature = 0.7f;
        config.echo = true;
        std::cout << "   ├── max_new_tokens: " << config.max_new_tokens << std::endl;
        std::cout << "   ├── temperature: " << config.temperature << std::endl;
        std::cout << "   └── echo: " << (config.echo ? "true" : "false") << " ✅" << std::endl;
        std::cout << std::endl;
        
        // 6. Run inference
        std::cout << "6. 🎯 Running multimodal inference..." << std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
        
        // Token callback - print tokens as they are generated
        std::function<void(const std::string&)> token_callback = [](const std::string& token) {
            std::cout << token << std::flush;
        };
        
        // Stats callback - print generation statistics
        std::function<void(const Stats&)> stats_callback = [](const Stats& stats) {
            std::cout << std::endl;
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            std::cout << std::endl;
            std::cout << "📊 Generation Statistics:" << std::endl;
            std::cout << "   ├── Generated tokens: " << stats.num_generated_tokens << std::endl;
            long inference_time = stats.inference_end_ms - stats.inference_start_ms;
            std::cout << "   ├── Total inference time: " << inference_time << "ms" << std::endl;
            if (inference_time > 0) {
                double tokens_per_sec = (stats.num_generated_tokens * 1000.0) / inference_time;
                std::cout << "   └── Tokens/second: " << std::fixed << std::setprecision(1) << tokens_per_sec << std::endl;
            }
        };
        
        // Generate
        auto generate_result = runner->generate(inputs, config, token_callback, stats_callback);
        if (generate_result != Error::Ok) {
            std::cerr << "Generation failed" << std::endl;
            return 1;
        }
        
        std::cout << std::endl;
        std::cout << "✅ Multimodal Runner Successfully Executed!" << std::endl;
        std::cout << std::endl;
        std::cout << "🔗 Implementation Details:" << std::endl;
        std::cout << "   └── Built with multimodal runner from commit 83749ae59d" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}