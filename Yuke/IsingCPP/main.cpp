//Ising model
//Author: Yuke Wang

#include <iostream>
#include "ising.hpp"
#include <chrono>

int main(int argc, const char * argv[]) {
    int iters = 10000;
    ising model(128,128,1,-1,0.1);
    ising model2(128,128,1,-1,0.1);
    auto startTime = std::chrono::system_clock::now();
    for (int i = 0; i < iters; i++){
        for (int j = 0; j < 1; j++){
            model2.step_optim();
            for (int k = 0; k < 2000; k++)
                model.step_walk();
        }
        model.draw("walk",1);
        model2.draw("optim",1);
    }
    auto endTime = std::chrono::system_clock::now();
    std::cout << "Random Walkers: " << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << std::endl;
    
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

