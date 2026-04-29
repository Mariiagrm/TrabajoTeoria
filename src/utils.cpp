
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <filesystem>
#include <omp.h>
#include <string>
#include <algorithm>
#include <stdio.h>

cv::Mat preprocessImage(const cv::Mat& img) {
    cv::Mat gray, blur, mask, result;

    // 1. Ensure image is grayscale
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }

    // 2. Gaussian Blur to remove floor texture noise (marble veins)
    cv::GaussianBlur(gray, blur, cv::Size(7, 7), 0);

    // 3. Inverse Otsu Thresholding
    cv::threshold(blur, mask, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // 4. Morphological Closing to fill small holes inside the object mask
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    // 5. Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
        return cv::Mat::zeros(img.size(), img.type());

    // 6. Keep only the largest contour (the bottle)
    double maxArea = 0;
    int maxAreaIdx = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) { maxArea = area; maxAreaIdx = (int)i; }
    }

    // 7. Clean mask with only the largest contour
    cv::Mat cleanMask = cv::Mat::zeros(mask.size(), CV_8UC1);
    cv::drawContours(cleanMask, contours, maxAreaIdx, cv::Scalar(255), cv::FILLED);

    // 8. Apply mask to original image
    cv::bitwise_and(gray, gray, result, cleanMask);
    return result;
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_carpeta>" << std::endl;
        return 1;
    }

    std::string path = argv[1];
    std::vector<std::string> imageFiles;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff")
            imageFiles.push_back(entry.path().string());
    }
    std::sort(imageFiles.begin(), imageFiles.end());

    int N = (int)imageFiles.size();
    int max_threads = omp_get_max_threads();
    std::cout << "Imagenes encontradas: " << N
              << " | Hilos disponibles: " << max_threads << std::endl;

    std::filesystem::create_directories("../data/clean_images");

    //  Versión SECUENCIAL 
    std::cout << "\n[Secuencial] Iniciando..." << std::endl;
    double t_sec_ini = omp_get_wtime();

    for (int i = 0; i < N; i++) {
        cv::Mat img = cv::imread(imageFiles[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Error cargando: " << imageFiles[i] << std::endl;
            continue;
        }
        cv::Mat processed = preprocessImage(img);
    }

    double t_sec = omp_get_wtime() - t_sec_ini;
    std::cout << "[Secuencial] Tiempo: " << t_sec << " s" << std::endl;

    //  Versión PARALELA 
    std::cout << "\n[Paralelo]   Iniciando con " << max_threads << " hilos..." << std::endl;
    double t_par_ini = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < N; i++) {
        cv::Mat img = cv::imread(imageFiles[i], cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            #pragma omp critical
            std::cerr << "Error cargando: " << imageFiles[i]
                      << " (hilo " << omp_get_thread_num() << ")" << std::endl;
            continue;
        }
        cv::Mat processed = preprocessImage(img);

        std::string outPath = "../data/clean_images/processed_" + std::to_string(i) + ".jpg";
        cv::imwrite(outPath, processed);
    }

    double t_par = omp_get_wtime() - t_par_ini;
    std::cout << "[Paralelo]   Tiempo: " << t_par << " s" << std::endl;

    //  Resumen 
    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Speedup: " << t_sec / t_par << "x  ("
              << max_threads << " hilos)" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;


    return 0;
}

