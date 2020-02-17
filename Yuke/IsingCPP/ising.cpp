//
//  ising.cpp
//  Ising
//
//  Created by Yuke Wang on 2/16/20.
//  Copyright © 2020 Yuke Wang. All rights reserved.
//

#include "ising.hpp"
ising::ising(){
}

//Randomly initilize matrix with -1 or 1
ising::ising(unsigned int x_size, unsigned int y_size){
    grid = arma::randi(x_size, y_size, arma::distr_param(0,1)) * 2 - 1;
}

ising::ising(unsigned int x_size, unsigned int y_size, float k, float j, float t){
    K = k;
    J = j;
    T = t;
    grid = arma::randi(x_size, y_size, arma::distr_param(0,1)) * 2 - 1;
}

void ising::resize(unsigned int x_size, unsigned int y_size){
    grid = arma::randi(x_size, y_size, arma::distr_param(0,1)) * 2 - 1;
}

//Convolute with Hamiltonian matrix
arma::fmat ising::deltaE() const{
    arma::fmat kernel({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}});
    arma::fmat fgrid = arma::conv_to<arma::fmat>::from(grid);
    return arma::conv2(fgrid , -2 * J * kernel, "same") % fgrid;
}

//Shift calculation for Energy. This is exceedingly slow
//arma::fmat ising::deltaEshift() const{
//    arma::fmat fgrid = arma::conv_to<arma::fmat>::from(grid);
//    arma::fmat sgrid = arma::shift(fgrid, 1, 0) + arma::shift(fgrid, -1, 0) + arma::shift(fgrid, 1, 1) + arma::shift(fgrid, -1, 1);
//    return arma::conv2(fgrid , -2 * J * sgrid, "same") % fgrid;
//}

void ising::step(){
    //arma::fmat delta = deltaE();
    //delta.print("Delta: ");
    arma::fmat prob = arma::exp(-deltaE()/(K * T));
    //prob.print("PROB: ");
    arma::umat flip = (arma::randu<arma::fmat>(grid.n_cols, grid.n_rows) < prob) % (arma::randu<arma::fmat>(grid.n_cols, grid.n_rows) < 0.1);
    grid = grid % (arma::conv_to<arma::imat>::from(flip)*(-2) + 1);
}

void ising::print(){
    grid.print("Matrix: ");
}

void ising::draw(){
    arma::mat image = arma::conv_to<arma::mat>::from(grid);
    const cv::Mat img(grid.n_cols, grid.n_rows, CV_64F, image.memptr());
    cv::namedWindow("Ising");
    cv::imshow("Ising", img);
    cv::waitKey(1);
}
