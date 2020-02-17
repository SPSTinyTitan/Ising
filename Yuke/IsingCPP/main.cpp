//Ising model
//Author: Yuke Wang

#include <iostream>
#include "ising.hpp"
#include <chrono>

int main(int argc, const char * argv[]) {
    ising model(1024,1024,1,-1,1.5);
    //model.draw();
    //cv::waitKey(0);
    auto startTime = std::chrono::system_clock::now();
    
    for (int i = 0; i < 100; i++){
        //for (int j = 0; j < 10; j++)
            model.step();
        //model.draw();
    }
    auto endTime = std::chrono::system_clock::now();

    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    return 0;
}

