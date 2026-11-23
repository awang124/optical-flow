# PLK / PFB
[[Link](https://awang124.github.io/preprint.pdf)] Wang, A. and Micheli, M. (2026). *Probabilistic Pyramidal Optical Flow via Iterated Extended Kalman Filtering over Image Resolutions*. arXiv.

### Abstract
Accurate motion perception and interpretation are central components of modern computer vision systems, with applications throughout science and engineering. Optical flow estimation is the mathematical computation of motion fields from sequential pairs of images in a video. Classical algorithms give deterministic point estimates, while deep learning models capable of uncertainty-aware estimation are data-intensive black boxes. The domain of structurally transparent probabilistic optical flow methods remains underdeveloped. We discuss two popular pyramidal optical flow algorithms—Lucas-Kanade and Farnebäck—and recognize that their recursion across image resolution levels parallels the recursion of the iterated extended Kalman filter (IEKF) traditionally across timesteps. We embed these algorithms within a modified IEKF to produce two novel probabilistic optical flow algorithms—Probabilistic Lucas-Kanade (PLK) and Probabilistic Farnebäck (PFB). For each of these, we develop three variants tailored to different assumptions; their respective measurement models are constructed using zeroth-order intensity constancy, first-order gradient constancy, and a combined formulation. We discuss an adaptive noise covariance estimation heuristic using measurement residuals for outlier and occlusion rejection. We evaluate our algorithms on the MPI-Sintel and KITTI-2015 benchmark datasets, and achieve substantial endpoint and F1 error reductions over classical baselines. Our methods also natively produce full flow covariance matrices, providing a geometrically interpretable quantification of estimation confidence and spatial conditioning.

### Datasets
[[Link](https://sintel.is.tue.mpg.de/)] **MPI-Sintel** (2012): adapted from the open-source computer-animated short film Sintel. It provides a training set, consisting of 1,041 image pairs with dense ground-truth flow fields, and a test set, of 552 image pairs and no ground-truth. All images are 8-bit RGB, which we convert to grayscale, and of resolution 436 × 1024. Test-set evaluation is conducted, i.e. final metrics are computed, via uploading test-set flow field estimates to the dataset’s website. The benchmark is known for featuring large displacements, non-rigid motions, atmospheric effects (e.g. fog, dust), frequent occlusions, fine structures, and independent object and camera motions. The dataset is provided in three render passes, of which we provide results for the latter two: *Albedo* obeys brightness constancy except at occlusions, *Clean* adds shading, shadows, specular reflections, and other lighting effects, and *Final* Adds motion blur, focus blur, and atmospheric effects.

[[Link](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)] **KITTI Stereo / Flow / Scene Flow** (2015): captured by a real-world camera on a moving vehicle in urban driving scenarios in Germany. It provides a training set, consisting of 200 image pairs with sparse ground-truth vectors computed using LiDAR, and a test set, of 200 image pairs and no groundtruth. All images are likewise 8-bit RGB, which we again convert to grayscale, and of resolution 375 × 1242. The benchmark features highly rigid motion, but also notably includes ego-motion (camera moves forward, environment shifts outward) and extreme lighting effects (e.g. sun glare, deep shadows, overexposed skies).

### Code Organization
`plk.py`: implements of Probabilistic Lucas-Kanade (Section 3.2)<br>
`pfb.py`: implements Probabilistic Farnebäck (Section 3.3)

The two files are largely identical outside core PLK / PFB IEKF algorithms, i.e. in functions for data loading, flow visualization, estimate evaluation, etc. They are left redundant to allow independent execution without requiring additional auxiliary files. All functions are mapped below.
```
PLK.py                     PFB.py (if different)           Description
├── IMPORTS
├── DATA LOADING
│   ├── readimg                                            load image (to grayscale)
│   ├── readMPI                                            load Sintel flow field
│   ├── readKITTI                                          load KITTI flow field
│   └── loadpair                                           load image pair & flow field
├── VISUALIZATION
│   ├── visimg                                             show image pair
│   ├── colorwheel                                         generate flow-color mapping
│   ├── computeColor                                       map flow vector ==> color
│   ├── flow2color                                         computeColor wrapper
│   └── visflo                                             show flow & covariance field
├── AUXILIARY FUNCTIONS ───┐
│   ├── gaussianPyramid    ├── gaussianPyramid             get Gaussian image pyramid
│   └── gaussianWeights    ├── gaussianWeights             get window weights
│                          ├── gramSchmidt                 orthogonalize FB polynomials
│                          └── generateFarnebackFilters    get FB coefficients
├── CORE ALGORITHMS ───────┐
│   ├── kernelI            ├── kernelI                     -intensity update GPU kernel
│   ├── PLKI               ├── PFBI                        -intensity IEKF algorithm
│   ├── kernelG            ├── kernelG                     -gradient update GPU kernel
│   ├── PLKG               ├── PFBG                        -gradient IEKF algorithm
│   ├── kernelC            ├── kernelC                     -combined update GPU kernel
│   └── PLKC               └── PFBC                        -combined IEKF algorithm
├── FLOW EVALUATION
│   ├── computeEPE                                         get flow field estimate EPE
│   ├── computeF1                                          get flow field estimate F1
│   └── run                                                collect dataset-wide EPE/F1
├── UNCERTAINTY EVALUATION
│   ├── computeAUC                                         get covariance field AUSE
│   └── run                                                collect dataset-wide AUSE/CC
└── PARAMETER OPTIMIZATION
    ├── searchParams                                       try parameters, collect EPE/F1
    ├── best                                               get best parameters via EPE/F1
    └── topk                                               get top-k parameters likewise
```
