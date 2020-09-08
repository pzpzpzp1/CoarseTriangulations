#include "simulator.h"

Simulator::Simulator(const char *imgPath, Imagem& img, ApproxType approxtype) {
    // get dimensions of image
    int maxX = img.width;
    int maxY = img.height;
    cout << "image is " << maxX << "x" << maxY << endl;
    if(approxtype == constant) {
        approx = new ConstantApprox(img, STARTING_STEP);
    } else if (approxtype == linear) {
        approx = new LinearApprox(img, STARTING_STEP);
    }

    //cout << "connecting to matlab... " << flush;
    //matlabPtr = startMATLAB();
    //cout << "done" << endl;

    // determine whether to use saliency map
    string saliencyString;
    bool salient = false;
    vector<string> yesAnswers = {"y", "Y"};
    cout << "Use saliency map for feature identification? y/N: ";
    // cin >> saliencyString;
    saliencyString = "n";
    // anything other than y/Y is false
    for(int i = 0; i < yesAnswers.size(); i++) {
        if(saliencyString == yesAnswers.at(i)) {
            salient = true;
        }
    }

    /* TODO: set saliency with matlab saliency map 
     * current code already connects to matlab,
     * change would just be changing matlab function call
     * and arguments */
    // assign pixel saliency values
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

    // determine initialization method
    string trimString;
    bool useTRIM = false; // default to uniform initialization
    cout << "Use TRIM initialization? y/N: ";
    //cin >> trimString;
    trimString = "n";
    // anything other than y/Y will be false
    for(int i = 0; i < yesAnswers.size(); i++) {
        if(trimString == yesAnswers.at(i)) {
            useTRIM = true;
        }
    }

    // initialize triangulation
    useTRIM = 0;
    if(useTRIM) { // get initial triangulation from matlab TRIM functions
        // prompt for density
        /*
        cout << "Density argument: ";
        cin >> density;
        if(cin.fail()) {
            cin.clear();
            cout << "defaulting to " << DENSITY_DEFAULT << endl;
            density = DENSITY_DEFAULT;
        }
        // add path to TRIM code
        vector<matlab::data::Array> genPathArgs({
            factory.createCharArray("../deps/trim")
        });
        auto generatedPath = matlabPtr->feval(u"genpath",genPathArgs);
        matlabPtr->feval(u"addpath", generatedPath);

        // read image
        vector<matlab::data::Array> pathToImage({
            factory.createCharArray(imgPath)
        });
        auto img = matlabPtr->feval(u"imread", pathToImage);

        // generate triangulation
        vector<matlab::data::Array> tArgs({
            img,
            factory.createScalar<double>(density) // density 
        });
        cout << "Triangulating...\n";
        vector<matlab::data::Array> output = matlabPtr->feval(u"imtriangulate", 3, tArgs);
        cout << "done\n";
        // vertices of triangulation
        matlab::data::Array vertices = output.at(0);
        matlab::data::Array triangleConnections = output.at(1);
        int n = vertices.getDimensions().at(0); // number of points in triangulation

        // initialize points of mesh
        cout << "Getting " << n << " points...\n";
        vector<Point> points;
        // appears to affect only large images:
        // adjust image boundaries to TRIM result (may crop a line of pixels)
        int minX = 1000;
        int minY = 1000;
        int maxX = 0;
        int maxY = 0;
        for(int i = 0; i < n; i++) {
            int x = vertices[i][0];
            int y = vertices[i][1];
            maxX = max(x, maxX);
            maxY = max(y, maxY);
            minX = min(x, minX);
            minY = min(y, minY);
        }
        for(int i = 0; i < n; i++) {
            // note these are 1-indexed pixel values; will need
            // to convert to usable points 
            int x = vertices[i][0];
            int y = vertices[i][1];
            // determine whether this point lies on edge of image
            bool isBorderX = (x == minX || x == maxX);
            bool isBorderY = (y == minY || y == maxY);
            Point p(x-0.5 - minX, y-0.5 - minY, isBorderX, isBorderY); // translate to coordinate system in this code
            points.push_back(p);
        }
        cout << "done\n";

        // convert connections to vector<vector<int>>
        cout << "Getting edges...\n";
        vector<array<int, 3>> edges;
        int f = triangleConnections.getDimensions().at(0); // number of triangles
        cout << "number of triangles: " << f << endl;
        for(int i = 0; i < f; i++) {
            array<int, 3> vertexInds;
            for(int j = 0; j < 3; j++) {
                // matlab is 1 indexed for some bizarre reason;
                // change back to zero indexing
                int ind = triangleConnections[i][j];
                vertexInds[j] = ind - 1;
            }
            edges.push_back(vertexInds);
        }
        cout << "done\n";
        cout << "Initializing mesh...\n";
        approx->initialize(points, edges);
        */
    } else {
        // prompt for dx
        cout << "Sample once every __ pixels? ";
        dx = DX_DEFAULT;
        //cin >> dx;
        if(cin.fail()) {
            cin.clear();
            dx = DX_DEFAULT;
            cout << "defaulting to " << DX_DEFAULT << endl;
        }
        approx->initialize(dx);
    }
    cout << "ready\n";
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