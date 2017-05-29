# GraduationProject

This project is a machine learning based system for translating Arabic Sign Language, or any sign language if it's trained for, into text, and when connected to a TTS engine to speech. It is the graduation project of us, Ali Annubani, Sara Yassin, and Ameer Alqam, from Birzeit University.

# Architecture

The proejct works by reading data from IMU sensors, specifically the Sparkfun LSM9DS, placed on the fingers of the user and controlled using an Arduino-compatible microcontroller. These IMU read Acceleration, Gryoscope, and Magnetometer readings, from which their active regions are extracted and translated into a feature-set., to be fed into a ML model for recognition.

# Structure

The software side of this project is composed of two main parts responsible for all the logic and processing fo the system, and one complementary part.

## DataReader

Contains all the Arduino code that drives the hardware of the system, it contains 3 subdirectories.

- `i2c_scanner_tca` contains a code that loops over all addresses of a Mux, and prints out all detected devices.
- `LSM9DS1` the main C++ based library for driving the IMUs.
- `LSM9DS1_Basic_I2C_one` performs the reading and transmission of the data of the IMUs.

## SystemPipeline

Contains all the processing aspects of the project. The system is composed as a funnel, or pipeline, with separate modules, each representing and independant stage, with intermediatery transformers to aid the data flow.

### `data`

Holds the raw collected data, as well as a basic script for reading the serial data sent by the Arduino.

### `features`

Contains all the code responsible for feature extraction, the main class here is the `Extractor` class, which is designed to be inherited with sub-classes having all the features required for the system, it automatically performs introspection and does the feature extraction given the appropriate data. Appropriate data being data compatible with the feature methods.

The feature set we used is encapsulated within the `FeaturesExtractor` class. To transform the data to a more vector friendly format, we have the `FeaturesTransformer` class holding this functionality.

In principle, to do feature extraction, a feature extraction class, or instacne, which extends from the main `Extractor` class is passed to the `FeaturesTransformer`, then the `FeaturesTransformer.transform` method is called on the required data.

### `learners`

Holds basic scripts for performing ML experiments.

### `segmentation`

Holds the  `Splicer` class, which is an implementation of our algorithm for detecting active regions and silent regions from the gesture signals.

### `utils`

Holds basic utilities shared accross the system, from a few decorators, to a dataset handling class.

## Coplementary component -- Data visualization

Contains a visualization of an IMU sensor, for real time demos, originally developed by Navio, and modified by us.
