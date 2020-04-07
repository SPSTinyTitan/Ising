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
#include <opencv2/core/opengl.hpp>
#include <vector>
#include <string>

class ising{
    
    
public:
    arma::imat grid;
    float K;            //Joules/Kelvin
    float J;            //Interaction Strength
    float T;            //Temperature
    arma::uvec update;
    arma::fmat prob;
    
    ising();
    ising(unsigned int x_size, unsigned int y_size);
    ising(unsigned int x_size, unsigned int y_size, float k, float j, float t);
    void resize(unsigned int x_size, unsigned int y_size);
    arma::fmat deltaE() const;
    arma::fmat deltaE_optim() const;
    void step();
    void step_optim();
    arma::uvec needs_update(arma::uvec ind);
    void print();
    void draw();

};


#endif /* ising_hpp */


