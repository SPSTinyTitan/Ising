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
#include <random>

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
    arma::fmat deltaE_update(arma::uvec pos) const;
    arma::fmat deltaE_shift() const;
    void step();
    void step_optim();
    arma::uvec needs_update(arma::uvec ind);
    arma::uvec needs_update_toroid(arma::uvec ind);
    void step_walk();
    void print();
    void draw(char* name, int waittime);
    

};


#endif /* ising_hpp */


