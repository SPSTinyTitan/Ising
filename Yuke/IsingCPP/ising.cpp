//
//  ising.cpp
//  Ising
//
//  Created by Yuke Wang on 2/16/20.
//  Copyright © 2020 Yuke Wang. All rights reserved.
//

#include "ising.hpp"
using namespace arma;



ising::ising(){
}

//Randomly initilize matrix with -1 or 1
ising::ising(unsigned int x_size, unsigned int y_size){
    grid = randi(x_size, y_size, distr_param(0,1)) * 2 - 1;
}

ising::ising(unsigned int x_size, unsigned int y_size, float k, float j, float t){
    K = k;
    J = j;
    T = t;
    grid = randi(x_size, y_size, distr_param(0,1)) * 2 - 1;
    update = linspace<uvec>(0, x_size * y_size - 1, x_size * y_size);
    prob = zeros<fmat>(x_size, y_size);
}

void ising::resize(unsigned int x_size, unsigned int y_size){
    grid = randi(x_size, y_size, distr_param(0,1)) * 2 - 1;
}

//Convolute with Hamiltonian matrix
fmat ising::deltaE() const{
    fmat kernel({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}});
    fmat fgrid = conv_to<fmat>::from(grid);
    return conv2(fgrid , -2 * J * kernel, "same") % fgrid;
}

//Shift calculation for Energy. This is exceedingly slow
//arma::fmat ising::deltaEshift() const{
//    arma::fmat fgrid = arma::conv_to<arma::fmat>::from(grid);
//    arma::fmat sgrid = arma::shift(fgrid, 1, 0) + arma::shift(fgrid, -1, 0) + arma::shift(fgrid, 1, 1) + arma::shift(fgrid, -1, 1);
//    return arma::conv2(fgrid , -2 * J * sgrid, "same") % fgrid;
//}

fmat ising::deltaE_optim() const{
    
    fmat kernel({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}});
    fmat fgrid = conv_to<fmat>::from(grid);
    return conv2(fgrid , -2 * J * kernel, "same") % fgrid;
}

void ising::step(){
    //fmat delta = deltaE();
    //delta.print("Delta: ");
    fmat prob = exp(-deltaE()/(K * T));
    //prob.print("PROB: ");
    umat flip = (randu<fmat>(grid.n_cols, grid.n_rows) < prob) % (randu<fmat>(grid.n_cols, grid.n_rows) < 0.1);
    grid = grid % (conv_to<imat>::from(flip)*(-2) + 1);
}

void ising::step_optim(){
    //fmat delta = deltaE();
    //delta.print("Delta: ");
    prob(update) = exp(-(deltaE()(update))/(K * T));
    //prob.print("PROB: ");
    umat flip = (randu<fmat>(grid.n_cols, grid.n_rows) < prob) % (randu<fmat>(grid.n_cols, grid.n_rows) < 0.1);
    uvec ind = find(flip == 1);
    grid(ind) = -grid(ind);
    update = needs_update(ind);
}

uvec ising::needs_update(uvec ind){
    uvec x = join_cols(ind - 1, ind + 1);
    uvec y = join_cols(ind - grid.n_rows, ind + grid.n_rows);
    uvec up = join_cols(x, y);
    up = up(find(up >= 0 && up <= (grid.n_rows * grid.n_cols - 1)));
    return join_cols(ind, up);
}

void ising::print(){
    grid.print("Matrix: ");
}

void ising::draw(){
    mat image = conv_to<mat>::from(grid);
    cv::Mat img(grid.n_cols, grid.n_rows, CV_64F, image.memptr());
    //const cv::ogl::Buffer texture(img);
    cv::namedWindow("Ising");
    cv::imshow("Ising", img);
    cv::waitKey(1);
}


