#pragma once

#include "IModule.h"
#include "Vision.h"

namespace Record
{
  class Parameters : public SwitchableParameters
  {
  public:
    Parameters() { Defaults(); }

    void Defaults()
    {
      _path = "";
      _manualTrigger = false;
    }

    bool Valid()
    {
      return true;
    }

    std::string GetPath() { return _path; }
    void SetPath(std::string path) { _path = path; }

    bool GetManualTrigger() { return _manualTrigger; }
    void SetManualTrigger(bool manualTrigger) { _manualTrigger = manualTrigger; }

  private:
    // path at which to save image stream.
    std::string _path;

    // set whether images are captured continuously or on demand.
    bool _manualTrigger;
  };

  class Data : public IData
  {
  public:
    void Clear() {}
    bool Valid()
    {
      return true;
    }

  private:

  };

  class Record : public IModule<Parameters, Vision::Data, Data>
  {
  public:

    Record(IParameters *params, IData *input, IData *output);

    void CreateFolder();

  private:
    void Process();
    void SaveToDisk();

    std::string _folderName;
    int _index = 0;
    bool _firstRun = true;
  };
}