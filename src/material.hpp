#pragma once

#include <string>
#include <vector>

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


};