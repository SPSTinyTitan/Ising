//
//  ising.cpp
//  Ising
//
//  Created by Yuke Wang on 2/16/20.
//  Copyright © 2020 Yuke Wang. All rights reserved.
//

#include "ising.hpp"
using namespace arma;

template<typename T>
T mod(T a, int n){return a - floor(a/n)*n;}

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

fmat ising::deltaE_optim() const{
    
    fmat kernel({{0, 1, 0}, {1, 0, 1}, {0, 1, 0}});
    fmat fgrid = conv_to<fmat>::from(grid);
    return conv2(fgrid , -2 * J * kernel, "same") % fgrid;
}

//Shift calculation for Energy.
fmat ising::deltaE_shift() const{
    fmat fgrid = conv_to<arma::fmat>::from(grid);
    fmat sgrid = shift(fgrid, 1, 0) + arma::shift(fgrid, -1, 0) + arma::shift(fgrid, 1, 1) + shift(fgrid, -1, 1);
    return -2 * J * fgrid % sgrid;
}

//fmat ising::deltaE_update(uvec pos) const{
//
//}

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
    prob(update) = exp(-(deltaE_shift()(update))/(K * T));
    //prob.print("PROB: ");
    umat flip = (randu<fmat>(grid.n_cols, grid.n_rows) < prob) % (randu<fmat>(grid.n_cols, grid.n_rows) < 0.1);
    uvec ind = find(flip == 1);
    grid(ind) = -grid(ind);
    update = join_cols(needs_update_toroid(ind), ind);
}


void ising::step_walk(){
    static imat dE = conv_to<imat>::from(deltaE_shift());
    static uint pos = grid.size()/2;
    static std::default_random_engine generator(0);
    static std::normal_distribution<float> distribution(0,10);
    static std::uniform_real_distribution<float> random(0,1);

    uint newposy = (pos % grid.n_rows + (int)distribution(generator) + grid.n_rows) % grid.n_rows;
    uint newposx = (pos / grid.n_rows + (int)distribution(generator) + grid.n_cols) % grid.n_cols;
    uint newpos = newposy + grid.n_rows * newposx;
//    std::cout << '(' << pos % grid.n_rows << ',' << pos/grid.n_rows << ')' << '\n';
//    std::cout << '(' << newpos % grid.n_rows << ',' << newpos/grid.n_rows << ')' << '\n';
    float oldprob = exp(-(dE(pos))/(K * T));
    float newprob = exp(-(dE(newpos))/(K * T));

    if (random(generator) < (newprob/oldprob)){
        pos = newpos;
        oldprob = newprob;
    }

    if (random(generator) < oldprob){
        //std::cout << pos % grid.n_rows << ',' << pos/grid.n_rows << std::endl;
        grid(pos) = -grid(pos);
        dE(pos) = -dE(pos);
        uvec ind(1);
        ind = pos;
        ind = needs_update_toroid(ind);
        dE(ind) -= 4 * J * grid(ind) * grid(pos);
    }
}

uvec ising::needs_update(uvec ind){
    uvec x = join_cols(ind - 1, ind + 1);
    uvec y = join_cols(ind - grid.n_rows, ind + grid.n_rows);
    uvec up = join_cols(x, y);
    up = up(find(up >= 0 && up <= (grid.n_rows * grid.n_cols - 1)));
    return up;
}

uvec ising::needs_update_toroid(uvec ind){
    uvec y = mod(ind, grid.n_rows);
    uvec x = ind/grid.n_rows;
    uvec a = join_cols(mod((uvec)(y + grid.n_rows - 1), grid.n_rows) + (x) * grid.n_rows,
                       mod((uvec)(y + 1), grid.n_rows) + (x) * grid.n_rows);
    uvec b = join_cols((y) + mod((uvec)(x + grid.n_cols - 1), grid.n_cols) * grid.n_rows,
                       (y) + mod((uvec)(x + 1), grid.n_cols) * grid.n_rows);
    a = join_cols(a,b);
    return a;
}

void ising::print(){
    grid.print("Matrix: ");
}

void ising::draw(char* name, int waittime){
    mat image = conv_to<mat>::from(grid);
    cv::Mat img(grid.n_cols, grid.n_rows, CV_64F, image.memptr());
    //const cv::ogl::Buffer texture(img);
    cv::namedWindow(name);
    cv::imshow(name, img);
    cv::waitKey(waittime);
}


