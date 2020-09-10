#include "simulator.h"

Simulator::Simulator(string imgPath, Imagem& img, ApproxType approxtype, int dxIn) {
    // get dimensions of image
    int maxX = img.width;
    int maxY = img.height;
    cout << "image is " << maxX << "x" << maxY << endl;
    if(approxtype == constant) {
        approx = new ConstantApprox(img, STARTING_STEP);
    } else if (approxtype == linear) {
        approx = new LinearApprox(img, STARTING_STEP);
    }

    // determine whether to use saliency map
    string saliencyString;
    bool salient = false;
    // TODO: set saliency with matlab saliency map 
    /*if(salient) { // for now this is just a Gaussian with standard dev 1/4 the diagonal length
        double stdev = 0.25 * sqrt(maxX * maxX + maxY * maxY);
        // matlab will generate a square filter
        int length = max(maxX, maxY);
        // pass arguments to matlab
        vector<matlab::data::Array> filterArgs({
            factory.createCharArray("gaussian"),
            factory.createScalar<double>(length),
            factory.createScalar<double>(stdev)
        });
        matlab::data::Array matlabMat = matlabPtr->feval(u"fspecial", filterArgs);
        vector<double> saliencyValues;
        // resulting structure is length x length, need to extract
        // a centered maxX x maxY matrix; do this by finding
        // the top left corner
        int startX = (length - maxX) / 2;
        int startY = (length - maxY) / 2;
        for(int i = 0; i < maxX; i++) {
            for(int j = 0; j < maxY; j++) {
                // scale the saliency up so that the sum of saliency values is approximately
                // maxX * maxY; consistent with default where every pixel has saliency 1
                double salience = (double) matlabMat[startX+i][startY+j] * maxX * maxY;
                saliencyValues.push_back(salience);
            }
        }
        approx->setSaliency(saliencyValues);
    }*/

    // initialize triangulation
    dx = dxIn;
    approx->initialize(dx);
    cout << "Simulator initialized.\n";
}

Simulator::~Simulator() {
    delete approx;
}

void Simulator::initialize() {
    // complete restart
    elapsedTimeVec.clear();
    energyVec.clear();
    errorVec.clear();
    //approx->registerMesh();
    registerMesh(approx);
    // setup gradient descent
    cout << "finding energy..." << endl;
    approxErr = approx->computeEnergy();
    newEnergy = approxErr + approx->regularizationEnergy();
    cout << "done, energy is " << newEnergy << endl;
    // initialize to something higher than newEnergy
    prevEnergy = newEnergy * 2;
    iterCount = 0;
    totalIters = 0;
    elapsedTimeVec.push_back(totalStep); // initial values
    errorVec.push_back(approxErr);
    energyVec.push_back(newEnergy);
}

void Simulator::step(double eps) {
    cout << "iteration " << iterCount << " (" << totalIters << " total)" << endl;
    // allow a fixed number of energy increases to avoid getting stuck
    totalStep += approx->step(prevEnergy, newEnergy, approxErr, (totalIters >= 10) && (iterCount >= 5));
    // data collection
    elapsedTimeVec.push_back(totalStep);
    errorVec.push_back(approxErr);
    energyVec.push_back(newEnergy);
    iterCount++;
    totalIters++;
    if(abs(prevEnergy - newEnergy) > eps * abs(prevEnergy)) {
        numSmallChanges = 0;
    } else {
        numSmallChanges++;
    }
    // handle display
    updateMesh(approx);
}

bool Simulator::step(int maxIter, double eps) {
    if(iterCount < maxIter && numSmallChanges < maxSmallChanges) {
        step(eps);
        return true;
    } 
    return false;
}

void Simulator::flow(int maxIter, double eps) {
    while(iterCount < maxIter && numSmallChanges < maxSmallChanges) {
        step(eps);
    } 
}

void Simulator::retriangulate(int num) {
    approx->subdivide(num);
    // re-initialize new mesh
    registerMesh(approx);
    // reset values
    iterCount = 0;
    numSmallChanges = 0;
    totalIters++;
    approxErr = approx->computeEnergy();
    newEnergy = approxErr + approx->regularizationEnergy();
    elapsedTimeVec.push_back(totalStep);
    energyVec.push_back(newEnergy);
    errorVec.push_back(approxErr);
    cout << "energy after subdivision: " << newEnergy << endl;
}

void Simulator::revealEdges(bool &display) {
    showEdges(approx, display);
}

void Simulator::cleanup() {
    // create suitable matlab arrays for data display purposes
    //matlab::data::TypedArray<int> iters = factory.createArray<int>({1, (unsigned long) totalIters + 1});
    //matlab::data::TypedArray<double> elapsedTime = factory.createArray<double>({1, (unsigned long) totalIters + 1});
    //matlab::data::TypedArray<double> energy = factory.createArray<double>({1, (unsigned long) totalIters + 1});
    //matlab::data::TypedArray<double> approxError = factory.createArray<double>({1, (unsigned long) totalIters + 1});
    if(elapsedTimeVec.size() == 0) return; // nothing was done
    /*for(int i = 0; i <= totalIters; i++) {
        iters[i] = i;
        elapsedTime[i] = elapsedTimeVec.at(i);
        energy[i] = energyVec.at(i);
        approxError[i] = errorVec.at(i);
    }
    matlabPtr->setVariable(u"x", iters);
    matlabPtr->setVariable(u"t", elapsedTime);
    matlabPtr->setVariable(u"E", energy);
    matlabPtr->setVariable(u"ae", approxError);
    // create and save matlab plots
    matlabPtr->eval(u"f=figure('visible', 'off'); plot(x, t); title('Elapsed Time')");
    matlabPtr->eval(u"exportgraphics(f, '../outputs/data_time.png')");
    matlabPtr->eval(u"g=figure('visible', 'off'); plot(x, E); axis([0 inf 0 inf]); title('Total Energy')");
    matlabPtr->eval(u"exportgraphics(g, '../outputs/data_energy.png')");
    matlabPtr->eval(u"h=figure('visible', 'off'); plot(x, ae); axis([0 inf 0 inf]); title('Approximation Error')");
    matlabPtr->eval(u"exportgraphics(h, '../outputs/data_error.png')");
    */

    // optionally convert screenshot sequences to video
    string confirm;
    cout << "run cleanup sequence? [WARNING: this is configured for Linux and will remove all .tga files in this directory] Y/n: " << flush;
    cin >> confirm;
    bool runCleanup = true;
    const vector<string> noAnswers = {"n", "N"};
    for(int i = 0; i < noAnswers.size(); i++) {
        if(confirm == noAnswers.at(i)) {
            runCleanup = false;
        }
    }
    if(runCleanup) {
        system("ffmpeg -hide_banner -loglevel warning -framerate 2 -i screenshot_%06d.tga -vcodec mpeg4 ../outputs/video_output.mp4");
        system("rm *.tga");
    }
}