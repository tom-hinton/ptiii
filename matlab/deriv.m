function [y,tOut] = deriv(x,tIn,varargin)
% deriv

% This is a template for creating a custom preprocessing function
% to be used in the Signal Analyzer App
%  x = a vector with input data
%  tIn = a vector with input time values. Expect an empty array for signals
%  in samples
%  y =  a vector with output data
%  tOut = a vector with output time values. Must be an empty array for
%  signals with no time information

y = gradient(x);
tOut = tIn;

