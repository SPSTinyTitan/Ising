//
//  ising.hpp
//  Ising
//
//  Created by Yuke Wang on 2/16/20.
//  Copyright © 2020 Yuke Wang. All rights reserved.
//

#ifndef ising_hpp
#define ising_hpp
#include <armadillo>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
class ising{
    
private:
    arma::imat grid;
    float K;            //Joules/Kelvin
    float J;            //Interaction Strength
    float T;            //Temperature

public:
    ising();
    ising(unsigned int x_size, unsigned int y_size);
    ising(unsigned int x_size, unsigned int y_size, float k, float j, float t);
    void resize(unsigned int x_size, unsigned int y_size);
    arma::fmat deltaE() const;
    void step();
    void print();
    void draw();

};


#endif /* ising_hpp */


