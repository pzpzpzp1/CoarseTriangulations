#include <iostream>
#include <fstream>
//#include "../deps/polyscope/include/polyscope/polyscope.h"
//#include "../deps/polyscope/include/polyscope/surface_mesh.h"
//#include "../deps/polyscope/include/polyscope/point_cloud.h"
//#include "../deps/polyscope/include/polyscope/view.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/point_cloud.h"
#include "polyscope/view.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "simulator.h"
#include "Imagem.h"

// using namespace matlab::engine;

static const double DENSITY_DEFAULT = 0.05; // experimentally a decent starting density
static const int DX_DEFAULT = 50; // sampling default

// let polyscope read values from point
double adaptorF_custom_accessVector2Value(const Point& p, unsigned int ind) {
    // reflect everything so that it displays correctly
    if (ind == 0) return -p.getX(); 
    if (ind == 1) return -p.getY();
    throw std::logic_error("bad access");
    return -1.;
}

// to aid in mesh registration, put constant triangle indexing here
const array<int, 3> triInds = {0,1,2};
const vector<array<int, 3>> singleTriangle = {triInds};
const array<double, 3> white = {1,1,1};
const vector<array<double, 3>> backgroundColor(8, white);

void registerMesh(Approx *approx) {
    if(approx->getApproxType() == constant) {
        auto triangulation = polyscope::registerSurfaceMesh2D("Triangulation", approx->getVertices(), approx->getFaces());
        auto colors = triangulation->addFaceColorQuantity("approximate colors", approx->getColors());
		// allow colors by default
  	    colors->setEnabled(true);
   	    // set material to flat to get more accurate rgb values
   	    triangulation->setMaterial("flat");
    } else if (approx->getApproxType() == linear) {
        cout << "registering mesh... " << flush;
        vector<Point> pts = approx->getVertices();
        vector<array<int, 3>> faces = approx->getFaces();
        vector<array<double, 3>> colors = approx->getColors();
        // register a separate mesh for each triangle
        for(int t = 0; t < faces.size(); t++) {
            vector<Point> thisTriangle; // hold vertices of faces.at(t)
            vector<array<double, 3>> vertexColors;
            for(int i = 0; i < 3; i++) {
                thisTriangle.push_back(pts.at(faces.at(t).at(i)));
                vertexColors.push_back(colors.at(3*t+i));
            }
            auto triangle = polyscope::registerSurfaceMesh2D(to_string(t), thisTriangle, singleTriangle);
            auto colorPtr = triangle->addVertexColorQuantity("linear approx", vertexColors);
            colorPtr->setEnabled(true);
            triangle->setMaterial("flat");
        }
        // create a background box so that the rendering centers
        auto bounding = polyscope::registerSurfaceMesh2D("bounding box", approx->boundingBox(), approx->boundingFaces());
        auto background = bounding->addFaceColorQuantity("background", backgroundColor);
        background->setEnabled(true);
        bounding->setMaterial("flat");
        cout << "done" << endl;
    }
}

void updateMesh(Approx *approx) {
    if(approx->getApproxType() == constant) {
        auto triangulation = polyscope::getSurfaceMesh("Triangulation");
        triangulation->updateVertexPositions2D(approx->getVertices());
        triangulation->addFaceColorQuantity("approximate colors", approx->getColors());
    } else if(approx->getApproxType() == linear) {
        vector<Point> pts = approx->getVertices();
        vector<array<int, 3>> faces = approx->getFaces();
        vector<array<double, 3>> colors = approx->getColors();
        for(int t = 0; t < faces.size(); t++) {
            vector<Point> thisTriangle;
            vector<array<double, 3>> vertexColors;
            for(int i = 0; i < 3; i++) {
                thisTriangle.push_back(pts.at(faces.at(t).at(i)));
                vertexColors.push_back(colors.at(3*t+i));
            }
            auto triangle = polyscope::getSurfaceMesh(to_string(t));
            triangle->updateVertexPositions2D(thisTriangle);
            triangle->addVertexColorQuantity("linear approx", vertexColors);
        }
    }
}

// if show, display all edges of mesh
// else hide all edges
void showEdges(Approx *approx, bool &show) {
    double edgeWidth = (show) ? 1 : 0;
    if(approx->getApproxType() == linear) {
        int numFaces = approx->getFaces().size();
        for(int t = 0; t < numFaces; t++) {
            auto triangle = polyscope::getSurfaceMesh(to_string(t));
            // default edge showing in black
            triangle->setEdgeColor({0,0,0});
            triangle->setEdgeWidth(edgeWidth);
        }
    } else if(approx->getApproxType() == constant) {
        polyscope::getSurfaceMesh("Triangulation")->setEdgeWidth(edgeWidth);
    }
    show = !show;
}

// in the linear case, highlight the edges of the t th triangle
void highlight(ApproxType approx, int t, bool on) {
    /*
    double edgeWidth = (on) ? 2 : 0;
    if (approx == linear) {
        auto triangle = polyscope::getSurfaceMesh(to_string(t));
        triangle->setEdgeColor({1,0,0}); // highlight in red
        triangle->setEdgeWidth(edgeWidth);
    }
    */
}

int main(int argc, char* argv[]) {
    // default image path and density
    const char *imgPath = "../images/flower.jpg";

    string inputPath = "../images/"; // to ensure non-null pointer later; find image directory
    if (argc >= 2) {
        inputPath += argv[1];
        imgPath = inputPath.c_str();
    }

    // CImg<unsigned char> image(imgPath);
    Imagem image(imgPath);

    int degree;
    cout << "Degree of approximation [0/1]: ";
    cin >> degree;
    if(cin.fail() || degree < 0 || degree > 1) {
        cin.clear();
        degree = 0;
        cout << "defaulting to constant approximation" << endl;
    }

    ApproxType approxtype = (degree == 0) ? constant : linear;

    // wrapper for running approximation
    Simulator sim(imgPath, image, approxtype);

    // set default values
    int maxIter = 100;
    int subdivisions = 20;
    double eps = 0.001;

    // lambda for displaying the starting triangulation
    auto initialize = [&]() {
       sim.initialize();
        // center mesh
        polyscope::view::resetCameraToHomeView();
        polyscope::resetScreenshotIndex();
        // screenshot
        polyscope::screenshot(true);
        polyscope::screenshot("../outputs/initial.tga", true);
    };

    // lambda for making a single step of gradient flow
    auto step = [&]() {
       if(sim.step(maxIter, eps)) {
            polyscope::screenshot(true);
       } else {
           polyscope::warning("done");
       }
    };

    // lambda for running the entire gradient flow
    auto runGradient = [&]() {
       while(sim.step(maxIter, eps)) {
           polyscope::screenshot(true);
       }
    };

    // lambda for retriangulating by subdivision
    auto retriangulate = [&]() {
        sim.retriangulate(subdivisions);
        polyscope::screenshot(true);
    };

    // lambda to handle GUI updates
    vector<string> angryButtonVec = {"MoRe TrIaNgLeS", "MORE TRIANGLES", "Are you done yet?", 
        "MORE", "I SAID MORE", "GIVE ME MORE", "We're done here.", "MORE",
        "Why are you so demanding?", "MORE", "I'm trying to be nice.", "MORE",
        "But you're really trying me here.", "MORE", "Okay, have it your way..", "More Triangles"};
    int numPresses = 0;
    string angryButton = "More Triangles";

    bool started = false; // whether process has been started and meshes registered
    bool displayEdges = true;

    auto callback = [&]() {
        ImGui::InputInt("max iterations", &maxIter); 
        ImGui::InputInt("# edges to divide", &subdivisions);
        ImGui::InputDouble("stopping condition", &eps);  

        if (ImGui::Button("Start")) {
            initialize();
            started = true;
        }
        ImGui::SameLine();
        // step by step
        if (ImGui::Button("Step") && started) {
            step();
        }
        ImGui::SameLine();
        // run the triangulation
        if (ImGui::Button("Gradient Flow") && started) {
            runGradient();
        }

        // allow retriangulation
        if (ImGui::Button(angryButton.c_str()) && started) {
            retriangulate();
            if(numPresses >= 3 && numPresses < 3 + angryButtonVec.size()) {
                angryButton = angryButtonVec.at(numPresses - 3);
            } else if (numPresses >= 3 + angryButtonVec.size()) {
                angryButton.append("!");
            }
            numPresses++;
        }

        if(ImGui::Button("Show Edges") && started) {
            sim.revealEdges(displayEdges);
        }
    };

    polyscope::init();
    polyscope::view::style = polyscope::view::NavigateStyle::Planar;
    polyscope::state::userCallback = callback;
    polyscope::show();
    if(started) {
        // take screenshot of just the triangulated image
        displayEdges = false;
        sim.revealEdges(displayEdges);
        polyscope::screenshot("../outputs/triangulation.tga", true);
        // take screenshot with edges shown
        displayEdges = true;
        sim.revealEdges(displayEdges);
        polyscope::screenshot("../outputs/edges.tga", true);
    }

    sim.cleanup();
	return 0;
}
