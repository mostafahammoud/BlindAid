#include "Functions.h"

const byte numChars = 23;
char receivedChars[numChars] = {'<', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '>'};
boolean newData = false;
int controlValues[7];
int analogPins[7] = { 3, 5, 6, 9, 10, 11 };

void InitializePins()
{
    for(int i = 0; i < 6; ++i)
        pinMode(analogPins[i], OUTPUT);
}

void ReceiveData()
{
    static boolean recvInProgress = false;
    static byte ndx = 0;
    char startMarker = '<';
    char endMarker = '>';
    char rc;

    while (Serial.available() > 0 && newData == false) {
        rc = Serial.read();

        if (recvInProgress == true) {
            if (rc != endMarker) {
                receivedChars[ndx] = rc;
                ndx++;
                if (ndx >= numChars) {
                    ndx = numChars - 1;
                }
            }
            else {
                receivedChars[ndx] = '\0'; // terminate the string
                recvInProgress = false;
                ndx = 0;
                newData = true;
            }
        }

        else if (rc == startMarker) {
            recvInProgress = true;
        }
    }
}

void ParseValues()
{
    String str(receivedChars);
    for(int i = 0; i < 7; ++i)
        controlValues[i] = str.substring(3*i, 3*i+3).toInt();
}

void ControlMotors()
{
    for(int i = 0; i < 5; ++i)
        analogWrite(analogPins[i] , controlValues[i]);

    newData = false;
}

void SendResponse()
{
    String response;
    for(int i = 0; i < 7; ++i)
        //response += String(controlValues[i]);
        Serial.print(controlValues[i]);

    char responseChar[21];
    response.toCharArray(responseChar, 21);

    //Serial.print(responseChar);
}
