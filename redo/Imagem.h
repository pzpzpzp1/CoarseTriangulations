#pragma once
#include <vector>
#include <iostream>
#include "../deps/lodepng/lodepng.h"
#include <fstream>

class Imagem
{
	public: 
		int width;
		int height;
		std::vector<unsigned char> data;

		Imagem(const char* filename);
		Imagem(int widtha, int heighta, std::vector<unsigned char> image);
		~Imagem();

		int get(int rgbind, int widthind, int heightind);
};

