#
# Copyright (c) 2018-2021, NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA Corporation is strictly prohibited.
#

from __future__ import print_function
from __future__ import absolute_import

import nvraw_v3
import array
import struct

class nvrawException(Exception):
    """
        nvrawException class
        this exception is raised when errors occur during nvrawV3 read operations
    """
    def __init__ (self, errorCode, msg = ""):
        try:
            self.value = errorCode
            self.msg = msg
        except Exception as inst:
            print("ERROR: {0}::{1}: {2} ".format(self.__class__.__name__,\
                 self.__class__.__init__.__name__, str(inst)))
            raise

    def __str__(self):
        return "ERROR: %s\nErrorCode: %s: %s" % \
                (self.msg, repr(self.value), nvraw_v3.getErrorString(self.value))

# TODO: Add err debug messages to self.funclogger.error in next patch
class NvRawFileV3(object):
    """
        NvRawFileV3 class
        this is a wrapper/utility class for the NvRaw_V3 class.
    """
    def __init__(self):
        # ----------------------------
        # These (permanent) data members will be loaded the whole time
        self._loaded = False
        self._nvrfUniqueObj = None
        self._filename = None
        self._nvrfReader = None

        self._versionHeaderReader = None
        self._baseHeaderReader = None
        self._planeHeaderReader = None

        self._width = 0
        self._height = 0
        self._frameCount = 0
        self._planeCount = 0

        self._embeddedDataLayout = nvraw_v3. EMBEDDED_DATA_LAYOUT_NOT_FOUND
        self._embeddedLineCountTop = 0
        self._embeddedLineCountBottom = 0
        self._bitsPerSample = 0
        self._pixelFormat = nvraw_v3.PIXEL_FORMAT_INT16
        self._fuseId = ""

        # LUT and its array count
        self._lutCount = 0
        self._pLUTfloat = array.array('f')
        self._pwl = nvraw_v3.PointFloatVector()

        # TODO: fix self._exposurePlaneVector to be a vector per frame in next patch
        self._exposurePlaneVector = []

        # ----------------------------
        # These (temporary) data members will be loaded depending on which frame(s)
        # the user specifies
        self._exposurePlaneReader = []
        self._frameDataReader = []
        self._pixelDataReader = []
        self._pixelDataArray = []
        self._pixelData = []
        self._maxPixelValue = []

        self._tempExposurePlaneReader = []
        self._tempFrameData = None
        self._tempFrameDataReader = []
        self._tempPixelData = None
        self._tempPixelDataReader = []
        self._tempPixelDataArray = []

        # Stores temporary data members in this list in the case the user specifies
        # more than 1 frame (range of frames)
        self._frameList = None
        self.className = self.__class__.__name__
        #=============================

    def readFileV3(self, filename):
        """!
            readFileV3()
            open the NvRaw_V3 image file and read/extract all the data from the
            metadata chunks and data chunks

            @param filename: a path to the NvRaw_V3 image file

            @return : return success if everything is properly read from the image file.
                      Otherwise, an exception call will be executed
        """
        try:
            self.cleanUp()
            err, nvrf3  = nvraw_v3.NvRawFileV3.openForReading(filename)
            if err:
                raise nvrawException(err, "Error while opening nvraw file for reading")
            self._nvrfUniqueObj = nvraw_v3.NvRawFileUniqueObj(nvrf3)
            self._loaded = True

            self._nvrfReader = nvraw_v3.INvRawFileReaderV1Cast(nvrf3)
            baseHeader = self._nvrfReader.getBaseHeader()

            self._baseHeaderReader = nvraw_v3.INvRawBaseHeaderReaderV1Cast(baseHeader)
            if (self._baseHeaderReader == None):
                raise nvrawException(nvraw_v3.NvError_InvalidState, "Couldn't get base header reader!")

            self._width = self._baseHeaderReader.getWidth()
            self._height = self._baseHeaderReader.getHeight()
            self._embeddedDataLayout = self._baseHeaderReader.getEmbeddedDataLayout()
            self._embeddedLineCountTop = self._baseHeaderReader.getEmbeddedLineCountTop()
            self._embeddedLineCountBottom =self._baseHeaderReader.getEmbeddedLineCountBottom()

            sensorInfo = self._nvrfReader.getSensorInfo()

            # TODO: Add in fuseId for next patch. SensorInfoReaderV1Cast function not found
            self._sensorInfoReader = nvraw_v3.INvRawSensorInfoReaderV1Cast(sensorInfo)
            if self._sensorInfoReader != None:
                self._fuseId = self._sensorInfoReader.getFuseId()
            else:
                self._fuseId = ""
                print("Warning: self._sensorInfoReader is None. self._fuseId not read")

            self._planeHeaderVector = nvraw_v3.ConstNvRawPlaneHeaderVector()
            self._nvrfReader.getPlaneHeaders(self._planeHeaderVector)

            self._numPlanes = self._planeHeaderVector.size()
            for i in range(self._planeHeaderVector.size()):
                self._planeHeaderReader = nvraw_v3.INvRawPlaneHeaderReaderV2Cast(self._planeHeaderVector[i])

                self._bitsPerSample = self._planeHeaderReader.getBitsPerSample()
                self._pixelFormat = self._planeHeaderReader.getPixelFormat()
                self._dynamicPixelBitDepth = self._planeHeaderReader.getDynamicPixelBitDepth()
                self._csiPixelBitDepth = self._planeHeaderReader.getCsiPixelBitDepth()
                self._planeHeaderReader.getPwlPoints(self._pwl)
                self._sensorModeType = self._planeHeaderReader.getSensorModeType()

            self._filename = filename
            return True
        except RuntimeError as err:
            raise RuntimeError("unable to read the file: {0}".format(err))

    def resetFramePointer(self, fileName):
        """!
            resetFramePointer()
            open the NvRaw_V3 image file and set the frame pointer

            @param fileName: a path to the NvRaw_V3 image file

            @return : return success if everything is properly read from the image file.
                      Otherwise, an exception call will be executed
        """
        self.closeFile()
        err, nvrf3  = nvraw_v3.NvRawFileV3.openForReading(fileName)
        self._nvrfUniqueObj = nvraw_v3.NvRawFileUniqueObj(nvrf3)
        if err:
            raise nvrawException(err, "Error while opening nvraw file for reading")

        self._nvrfReader = nvraw_v3.INvRawFileReaderV1Cast(nvrf3)
        return True

    def jumpToFrame(self, frameNum):
        """!
            jumpToFrame()
            go to the Nth frame from the current frame

            @param frameNum: Nth number of frame from the current frame

            @return : raise an exception if an error occured
        """
        self._frameList = nvraw_v3.ConstNvRawFrameUniqueObjVector()

        # traverses internal pointer to frameNumStart
        # with default parameters, this loop will do nothing. pinter will still be
        # at frame 0
        for i in range(frameNum):
            err = self._nvrfReader.getNextFrames(self._frameList.getAddr(), 1)
            if err:
                raise nvrawException(err, "Error while reading frames")

    def loadFrames(self, numFrames):
        """!
            loadFrames()
            load the Nth frame from the current frame

            @param numFrames: load x number of frames

            @return : raise an exception if an error occured
        """
        # Load in the actual number of frames desired
        err = self._nvrfReader.getNextFrames(self._frameList.getAddr(), numFrames)
        if err:
            raise nvrawException(err, "Error while reading frames")

    def loadFrameReader(self, frameNum):
        """!
            loadFrameReader()
            load the FrameReader instance at Nth frame. It will also reset the
            exposure plane from the new FrameReader instance

            @param frameNum: Nth number of frame from the current frame

            @return : raise an exception if an error occured
        """
        frameReader = nvraw_v3.INvRawFrameReaderV1Cast(self._frameList.getValueAtIndex(frameNum).get())

        if (frameReader == None):
            raise nvrawException(nvraw_v3.NvError_InvalidState, "Couldn't get frame reader!")

        self._exposurePlaneVector = nvraw_v3.ConstNvRawExposurePlaneVector()
        err = frameReader.getExposurePlanes(self._exposurePlaneVector)
        if err:
            raise nvrawException(err, "Error while getting exposurePlanes")

    def loadExposurePlanes(self, planeNum):
        """!
            loadExposurePlanes()
            obtain the exposure plane reader from the selected plane number of the
            current frame

            @param planeNum: selected plane number from the current plane list

            @return : raise an exception if an error occured
        """
        try:
            self._tempExposurePlaneReader.append(
                nvraw_v3.INvRawExposurePlaneReaderV1Cast(self._exposurePlaneVector[planeNum]))
        except RuntimeError as err:
            raise RuntimeError("unable to get the exposurePlane[{0}]: {1}".format(planeNum, err))

    def loadFrameData(self, planeNum):
        """!
            loadFrameData()
            obtain the frame data from the selected exposure plane

            @param planeNum: selected plane number from the current plane list

            @return : raise an exception if an error occured
        """
        try:
            self._tempFrameData = self._tempExposurePlaneReader[planeNum].getFrameData()

            self._tempFrameDataReader.append(
                nvraw_v3.INvRawFrameDataReaderV1Cast(self._tempFrameData))
        except RuntimeError as err:
            raise RuntimeError("unable to load frame data from plane[{0}]: {1}".format(planeNum, err))

    def loadPixelData(self, planeNum):
        """!
            loadPixelData()
            obtain the pixel data from the selected exposure plane

            @param planeNum: selected plane number from the current plane list

            @return : raise an exception if an error occured
        """
        try:
            self._tempPixelData = self._tempExposurePlaneReader[planeNum].getPixelData()

            if (self._tempPixelData == None):
                raise nvrawException(nvraw_v3.NvError_InvalidState, "Couldn't get pixel reader!")

            self._tempPixelDataReader.append(
                nvraw_v3.INvRawPixelDataReaderV1Cast(self._tempPixelData))

            if (self._tempPixelDataReader[planeNum] == None):
                raise nvrawException(err, "Error with getting PixelData reader!")

            pixelDataBlob = nvraw_v3.cdata(self._tempPixelDataReader[planeNum].getPixelData(),
                                            self._tempPixelDataReader[planeNum].getSize())
            self._tempPixelDataArray.append(array.array("h"))
            self._tempPixelDataArray[planeNum].fromstring(pixelDataBlob)
        except RuntimeError as err:
            raise RuntimeError("not able to load pixel data from plane[{0}]: {1}".format(planeNum, err))

    def loadNvraw(self, frameNumStart = 0, numFrames = 1):
        """!
            loadNvraw()
            set the plane pointer and frame pointer of the selected frame and
            also read the pixel data array from the selected plane

            @param frameNumStart : starting frame of the current nvraw_v3 file
            @param numFrames : number of frames to be loaded into FrameDataReader

            @return : raise an exception if an error occured
        """
        del self._exposurePlaneReader[:]
        del self._frameDataReader[:]
        del self._pixelDataReader[:]
        del self._pixelDataArray[:]
        del self._pixelData[:]
        del self._maxPixelValue[:]

        self.jumpToFrame(frameNumStart)
        self.loadFrames(numFrames)

        for i in range(self._frameList.size()):
            self.loadFrameReader(i)

            del self._tempExposurePlaneReader[:]
            self._tempFrameData = None
            del self._tempFrameDataReader[:]
            self._tempPixelData = None
            del self._tempPixelDataReader[:]
            del self._tempPixelDataArray[:]

            for j in range(self._exposurePlaneVector.size()):
                self.loadExposurePlanes(j)
                self.loadFrameData(j)
                self.loadPixelData(j)

            self._exposurePlaneReader.append(self._tempExposurePlaneReader)
            self._frameDataReader.append(self._tempFrameDataReader)
            self._pixelDataReader.append(self._tempPixelDataReader)
            self._pixelDataArray.append(self._tempPixelDataArray)
        self.closeFile()

        #pre-allocate self_.pixelData
        for i in range(0, len(self._pixelDataArray)):
            self._pixelData.append([])
            self._maxPixelValue.append([])
            for j in range(0, len(self._pixelDataArray[i])):
                self._pixelData[i].append([])
                self._maxPixelValue[i].append([])
        return self.adjustPixelDataV3()

    def closeFile(self):
        """!
            closeFile()
            close the current frame pointer

            @return : raise an exception if an error occured
        """
        try:
            self._nvrfUniqueObj.get().close()
        except RuntimeError as err:
            raise RuntimeError("unable to close a file: {0}".format(err))

    def cleanUp(self):
        """!
            cleanup()
            reset pre-allocated element to default stage and free the memory

            @return : raise an exception if an error occured
        """
        try:
            del self._exposurePlaneReader[:]
            del self._frameDataReader[:]
            del self._pixelDataReader[:]
            del self._pixelDataArray[:]
            del self._pixelData[:]

            del self._tempExposurePlaneReader[:]
            self._tempFrameData = None
            del self._tempFrameDataReader[:]
            self._tempPixelData = None
            del self._tempPixelDataReader[:]
            del self._tempPixelDataArray[:]

            self._nvrfUniqueObj = None
            self._filename = None
            self._nvrfReader = None

            self._versionHeaderReader = None
            self._baseHeaderReader = None
            self._planeHeaderReader = None
        except RuntimeError as err:
            print("not able to clean up data\n%s"%str(err))

    def generateLUTfloat(self, normalizedOutput = True):
        # convert PWL to pLUTfloat
        if (0 == self._pwl.size()):
            print("ERROR: cannot converting PWL to LUT; PWL is empty\n")
            return False
        lutVectorFloat = nvraw_v3.FloatVector()
        nvraw_v3.computePwlTable(self._pwl, self._bitsPerSample, self._dynamicPixelBitDepth, normalizedOutput, lutVectorFloat)
        pLUTsize = (1 << self._bitsPerSample)
        if (lutVectorFloat.size() != pLUTsize) :
            print("ERROR: LUT size {0} != max value of bitPerSample {1}".format(lutVectorFloat.size(),self._lutCount))
            return False
        del self._pLUTfloat[:]
        for i in range(0,pLUTsize):
            self._pLUTfloat.append(lutVectorFloat[i])
        return True

    def isCompressed(self):
        """!
            isCompressed()
            check if the data set is in compressed format

            @return : return true if dynamic bit depth is greater than CSI bit depth
        """
        res = False
        if self._dynamicPixelBitDepth > self._csiPixelBitDepth:
            res = True
        return res

    def getPeakPixelValueV3(self):
        """
            getPeakPixelValueV3()
            Returns the peak pixel or maximum possible value based on the dynamic bit depths.
            This is different than maxValue, which is the highest value actually captured
            in image and must be less than peakValue.

            @return :  2^(dynamicBitDepth)-1 in float format
        """
        peakValue = (1 << self._dynamicPixelBitDepth) - 1

        return float(peakValue)

    def adjustPixelDataV3(self):
        """
            adjustPixelDataV3()
            remove the embedded top/bottom data if it's not already stripped from
            the pixel data array. In addition, if the pixel data is in compressed format,
            the method will decompress the data

        """
        try:
            self._pixelDataConverted = False
            for i in range(0, len(self._pixelDataArray)):
                for j in range(0, len(self._pixelDataArray[i])):
                    if (nvraw_v3.EMBEDDED_DATA_LAYOUT_STRIPPED != self._embeddedDataLayout) and\
                    (nvraw_v3.EMBEDDED_DATA_LAYOUT_NOT_FOUND != self._embeddedDataLayout) and\
                    (nvraw_v3.EMBEDDED_DATA_LAYOUT_UNKNOWN != self._embeddedDataLayout):
                        """Adjust pixel data by taking into account embedded lines at the top
                        and bottom
                        """
                        # print "Original: Width " + str(self._width) + " Height " + str(self._height) + \
                        #       " embLineCountTop " + str(self._embeddedLineCountTop) + \
                        #       " embLineCountBottom " + str(self._embeddedLineCountBottom) + \
                        #       " size of _pixelDataArray " + str(len(self._pixelDataArray))
                        self._embeddedLineCountTop = self._baseHeaderReader.getEmbeddedLineCountTop()
                        self._embeddedLineCountBottom = self._baseHeaderReader.getEmbeddedLineCountBottom()

                        if ( (nvraw_v3.EMBEDDED_DATA_LAYOUT_T_D_B == self._embeddedDataLayout) or \
                            (nvraw_v3.EMBEDDED_DATA_LAYOUT_T_B_D == self._embeddedDataLayout) or \
                            (nvraw_v3.EMBEDDED_DATA_LAYOUT_T_D == self._embeddedDataLayout) ):

                            if ( (nvraw_v3.EMBEDDED_DATA_LAYOUT_T_D_B == self._embeddedDataLayout) or \
                                (nvraw_v3.EMBEDDED_DATA_LAYOUT_T_D == self._embeddedDataLayout) ):

                                # print "Copying Pixel data: From %u to %u" % (((self._width * self._embeddedLineCountTop)), \
                                # ((self._height - self._embeddedLineCountBottom) * self._width));
                                self._pixelData[i][j] = self._pixelDataArray[i][j][(self._width * self._embeddedLineCountTop) : \
                                            ((self._height - self._embeddedLineCountBottom) * self._width)]

                            elif (nvraw_v3.EMBEDDED_DATA_LAYOUT_T_B_D == self._embeddedDataLayout) :
                                # print "Copying Pixel data: From %u to %u" % (((self._width * self._embeddedLineCountTop)), \
                                # ((self._height - self._embeddedLineCountBottom) * self._width));
                                self._pixelData[i][j] = self._pixelDataArray[i][j][self._width * (self._embeddedLineCountTop + self._embeddedLineCountBottom) : \
                                                (self._height * self._width)]
                        elif (nvraw_v3.EMBEDDED_DATA_LAYOUT_D_B == self._embeddedDataLayout) :
                            # print "Copying Pixel data: From %u to %u" % (((self._width * self._embeddedLineCountTop)), \
                            # ((self._height - self._embeddedLineCountBottom) * self._width));
                            self._pixelData[i][j] = self._pixelDataArray[i][j][0 : \
                                                (self._height - self._embeddedLineCountBottom) * self._width]

                        self._height = self._height - (self._embeddedLineCountTop + self._embeddedLineCountBottom)
                        #print "\t Width " + str(self._width) + " Height " + str(self._height)
                    else:
                        self._pixelData[i][j] = self._pixelDataArray[i][j]

                    # Now uncompress the data if it is compressed
                    if ( self.isCompressed() ):
                        # Decompressed output data will be in self._pixelDataArray
                        if (0 == self._pwl.size()):
                            print("ERROR: cannot decompressing {0}; the PWL is empty".format(self._filename))
                            return False
                        elif (False == self.generateLUTfloat()):
                            print("ERROR: converting from PWL to LUT failed")
                            return False

                        pixelDataF32 = array.array('f', self._pixelData[i][j])
                        # map() from unsigned short to float had issues and would result
                        # in error in swig function during the call to NvRawFile_decompress_direct
                        # Hence using a loop to copy each pixel.
                        # pixelDataF32 = map(float, self._pixelDataArray)

                        nvraw_v3.NvRawFile_decompress_direct(self._pLUTfloat, \
                                                    self._height, self._width,   \
                                                    self._pixelData[i][j], self._csiPixelBitDepth, \
                                                    pixelDataF32, self._pixelFormat)

                        # The pixel values will be normalized to 1 (values from 0.0000 to 1.0000) in the function
                        # NvRawFile_decompress_direct. We need to scale these values back to the bitDepth
                        # Ex: For IMX185 Mode 4 (12 bit combined compressed), Dynamic bit depth is 16
                        # and CSI bit depth is 12. getPeakPixelValue() below will return 2^16 -1 = 65535
                        # Pixel values here will be scaled from the range of 0.0000 to 1.0000 to 0.0 to 65535.0
                        #
                        peakValue = self.getPeakPixelValueV3()

                        # Redefine the array as the long type this time. After decompression, integer values are bigger
                        # and we need to fit these back into the integer space holders.
                        self._pixelData[i][j] = array.array('L')

                        maxValue = 0.0
                        maxValueScaled = 0.0
                        for k in range(0, self._width * self._height, 1):
                            val = pixelDataF32[i]
                            if (val > maxValue):
                                maxValue = val

                            val = pixelDataF32[k] * peakValue
                            self._pixelData[i][j].append(int(val))
                            if (val > maxValueScaled):
                                maxValueScaled = val

                        self._maxPixelValue[i][j]= int(maxValueScaled)
                        self._pixelDataConverted = True

                        # Set the pixel format to Int16 from here onwards
                        self._pixelFormat = nvraw_v3.PIXEL_FORMAT_INT16

                # print "adjustPixelData: peakValue %u maxValue %4.4f maxValueScaled %4.4f" % \
                #            (peakValue, maxValue, maxValueScaled)
                # print "nvrawfile.py: pixelDataF32 After multiplying with bitDepth: maxValue %u " % (maxValue)
                # print ("%03.4f %03.4f %03.4f %03.4f %03.4f %03.4f %03.4f %03.4f " % (pixelDataF32[0], pixelDataF32[1], pixelDataF32[2], pixelDataF32[3], pixelDataF32[4], pixelDataF32[5], pixelDataF32[6], pixelDataF32[7]))
            return True
        except RuntimeError as err:
            print("ERROR: not able to adjust pixel value::{0}".format(str(err)))
            return False


    def getActualBitsPerSample(self):
        """ Returns the bit depth
        """
        if (self._dynamicPixelBitDepth > 0):
           return self._dynamicPixelBitDepth
        return self._bitsPerSample
