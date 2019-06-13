{\rtf1\ansi\ansicpg1252\cocoartf1561\cocoasubrtf600
{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\froman\fcharset0 Times-Roman;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue233;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c93333;}
\paperw11900\paperh16840\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 The MATLAB program in this folder performs: \
Training of a Convolutional Neural Network and classification of test set\
\
Instructions:\
Please make sure <MatConvNet> package is downloaded and unpacked into current directory.\
Package library must be compiled and installed\
If not, please follow instructions from:\
\pard\pardeftab720\sl280\partightenfactor0
{\field{\*\fldinst{HYPERLINK "http://www.vlfeat.org/matconvnet/install/"}}{\fldrslt 
\f1 \cf2 \expnd0\expndtw0\kerning0
\ul \ulc2 http://www.vlfeat.org/matconvnet/install/}}
\f1 \cf2 \expnd0\expndtw0\kerning0
\ul \ulc2 \
\
\pard\pardeftab720\sl280\partightenfactor0

\f0 \cf0 \kerning1\expnd0\expndtw0 \ulnone Please copy folder and paste it in /matconvnet-1.0-beta25/examples/\
\
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0
\cf0 To train the CNN please run the main file:\
/matconvnet-1.0-beta25/examples/my_face_recog/cnn_cmu.m\
\
Note:\
Training will only occur if the results folder is empty, otherwise it will display results from previous trainings.\
To allow retraining of CNN, please go to folder: /matconvnet-1.0-beta25/data/cmu-baseline-simplenn\
and delete all its contents before running main file }