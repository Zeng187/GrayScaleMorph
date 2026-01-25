#pragma once

#include <string>
#include <vector>
#include "common.hpp"
//#include <Eigen/core>

class Grayscale_Material
{
public:
	Grayscale_Material(const std::string& filePath);
	~Grayscale_Material(){};

	std::string name="";
	std::string description = "";
	
	std::vector<double> youngs_modulus;
	std::vector<double> strech_ratio;
	int count =0;

	double thickness = 1.0;

};

class ActiveComposite: public Grayscale_Material
{
public:
	ActiveComposite(const std::string& filePath);

	std::vector<double> lambda;
	std::vector<double> kappa;
	std::vector<double> E_moduls;
	double2 range_lam;
	double2 range_kap;


};