#pragma once

#include "IDetect.h"
#include "Capture.h"

namespace Vision
{
  namespace TrafficLight
  {
    class Result : public IResult
    {
    public:
      enum Color { Red, Green, Yellow, None };

      Result() { _radius = 0; }
      Result(cv::Point center, float radius, Color color) { _center = center; _radius = radius; _color = color; _count = 1; }
      Result(Color color, float confidence[3]) { for (int i = 0; i < 3; ++i) _confidence[i] = confidence[i]; }

      void Clear() { _center = cv::Point(0, 0); _radius = 0; _color = Red; }

      float CartesianDistance(Result &c2) { return (float)cv::norm(_center - c2._center); }
      float RadiusDifference(Result &c2) { return abs(_radius - c2._radius); }
      bool SameColor(Result &c2) { return _color == c2._color; }

      cv::Point _center;
      float _radius;
      Color _color;
      int _count = 0;
      float _confidence[3];
    };

    class Data : public IData
    {
    public:
      Data() { _results.push_back(Result(cv::Point(0, 0), 10, Result::Color::None)); }
      void Clear() { _results.clear(); }

      bool Valid()
      {
        return true;
      }

      std::vector<Result> Get() { return FilterByConsecutiveCount(); }
      Result::Color GetColor() { if (_results.size() > 0 && _results.at(0)._count > _consecutiveCount) return _results.at(0)._color; else return Result::Color::None; }
      // for blob detector version
      void Set(std::vector<Result> &results) { MatchPoints(results); }
      // for deep learning version
      void Set(Result::Color color, float confidence[3])
      {
        if (_results.at(0)._color != color)
          _results.at(0)._count = 0;

        for (int i = 0; i < 3; ++i)
          _results.at(0)._confidence[i] = confidence[i];

        _results.at(0)._color = color;
        _results.at(0)._count++;
      }

      int Size() { return (int)_results.size(); }
      Result& At(int i) { return _results.at(i); }
      void SetParams(int consecutiveCount, int maximumDistance, int maximumRadiusDiff) { _consecutiveCount = consecutiveCount; _maximumDistance = maximumDistance; _maximumRadiusDiff = maximumRadiusDiff; }

      std::vector<Result> FilterByConsecutiveCount();
      void MatchPoints(std::vector<Result> &results);

    private:
      std::vector<Result> _results;

      int _consecutiveCount;
      int _maximumDistance;
      int _maximumRadiusDiff;
    };

    class Base : public IDetect<Parameters, Capture::Data, Data>
    {
    public:
      Base(IParameters *params, IData *input, IData *output);
      static Base *Base::MakeTrafficLight(IParameters *params, IData *input, IData *output);

    protected:

    };
  }
}