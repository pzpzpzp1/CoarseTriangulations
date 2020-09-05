#include "Imagem.h"
#include <vector>

Imagem::Imagem(const char* filename) {
    unsigned widtha, heighta;

    //decode
    unsigned error = lodepng::decode(data, widtha, heighta, filename);

    //if there's an error, display it
    if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    //the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
    /* for (int i = 0; i < data.size(); i++) {
        //std::cout << " " << image[i] << " ";
        std::printf(" %d ", image[i]);
    } */
}

Imagem::Imagem(int widtha, int heighta, std::vector<unsigned char> image) {
	width = widtha;
	height = heighta;
	data = image; // 4 x width x height
}

Imagem::~Imagem() {
}

int Imagem::get(int rgbind, int widthind, int heightind) {
	int index = 4 * (heightind * width + widthind) + rgbind;
	auto val = data[index];
	return (int)val;
}