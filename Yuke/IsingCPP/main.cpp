//Ising model
//Author: Yuke Wang

#include <iostream>
#include "ising.hpp"
#include <chrono>

int main(int argc, const char * argv[]) {
    int iters = 4000;
    ising model(512,512,1,-1,1);
    auto startTime = std::chrono::system_clock::now();
    for (int i = 0; i < iters; i++){
        for (int j = 0; j < 1; j++)
            model.step_optim();
        model.draw();
    }
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Optimized: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
    
//    ising model2(256,256,1,-1,1.5);
//
//    startTime = std::chrono::system_clock::now();
//    for (int i = 0; i < iters; i++){
//        for (int j = 0; j < 1; j++)
//            model2.step_optim();
//        //model.draw();
//    }
//    endTime = std::chrono::system_clock::now();
//    std::cout << "Optimized: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
//
    return 0;
}

