//----------------------------------------------------------------------------------
// File:        NvAppBase/NvSampleApp.cpp
// SDK Version: v3.00 
// Email:       gameworks@nvidia.com
// Site:        http://developer.nvidia.com/
//
// Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//----------------------------------------------------------------------------------
#include "NvSampleApp.h"
#include "NvLogs.h"
#include "NvImage.h"
#include "NvString.h"
#include "NvTokenizer.h"

#include "NsAllocator.h"
#include "NsIntrinsics.h"

#include <stdarg.h>
#include <sstream>

NvSampleApp::NvSampleApp() :
	NvAppBase()
	, mFrameDelta(0.0f)
	, mEnableFPS(true)
	, m_desiredWidth(0)
	, m_desiredHeight(0)
	, mTestMode(false)
	, mTestDuration(0.0f)
	, mTestRepeatFrames(1)
    , mUseRenderThread(false)
    , mRenderThreadRunning(false)
    , mThread(NULL)
	, m_testModeIssues(TEST_MODE_ISSUE_NONE)
	, mEnableInputCallbacks(true)
	, mUseFBOPair(false)
    , m_fboWidth(0)
    , m_fboHeight(0)
	, mLogFPS(false)
	, mTimeSinceFPSLog(0.0f)
{
    mFrameTimer = createStopWatch();

    mEventTickTimer = createStopWatch();

    mAutoRepeatTimer = createStopWatch();
    mAutoRepeatButton = 0; // none yet! :)
    mAutoRepeatTriggered = false;

    const std::vector<std::string>& cmd = getCommandLine();
    std::vector<std::string>::const_iterator iter = cmd.begin();

    while (iter != cmd.end()) {
        if (0==(*iter).compare("-w")) {
            iter++;
            std::stringstream(*iter) >> m_desiredWidth;
        } else if (0==(*iter).compare("-h")) {
            iter++;
            std::stringstream(*iter) >> m_desiredHeight;
        } else if (0==(*iter).compare("-testmode")) {
            mTestMode = true;
            iter++;
            std::stringstream(*iter) >> mTestDuration;
            iter++;
            mTestName = (*iter); // both std::string
        } else if (0 == (*iter).compare("-repeat")) {
            iter++;
            std::stringstream(*iter) >> mTestRepeatFrames;
        } else if (0 == (*iter).compare("-fbo")) {
            mUseFBOPair = true;
            iter++;
            std::stringstream(*iter) >> m_fboWidth;
            iter++;
            std::stringstream(*iter) >> m_fboHeight;
		} else if (0 == (*iter).compare("-logfps")) {
			mLogFPS = true;
		}
		
        iter++;
    }

    nvidia::shdfnd::initializeNamedAllocatorGlobals();
    mThread = NULL;
    mRenderSync = new nvidia::shdfnd::Sync;
    mMainSync = new nvidia::shdfnd::Sync;
}

NvSampleApp::~NvSampleApp() 
{ 
    // clean up internal allocs
    delete mFrameTimer;
    delete mEventTickTimer;
    delete mAutoRepeatTimer;

    //delete m_transformer;
}

bool NvSampleApp::baseInitRendering(void) {
	if (!platformInitRendering())
		return false;
    initRendering();

	return true;
}

void NvSampleApp::baseInitUI(void) {
}

void NvSampleApp::baseReshape(int32_t w, int32_t h) {
    getAppContext()->platformReshape(w, h);

    if ((w == m_width) && (h == m_height))
        return;

    m_width = w;
    m_height = h;

    reshape(w, h);
}

void NvSampleApp::baseUpdate(void) {
    update();
}

void NvSampleApp::baseDraw(void) {
    draw();
}

void NvSampleApp::baseDrawUI(void) {
    drawUI();
}

bool NvSampleApp::characterInput(uint8_t c) {
    // In on-demand rendering mode, we trigger a redraw on any input
    if (mPlatform->getRedrawMode() == NvRedrawMode::ON_DEMAND)
        mPlatform->requestRedraw();

    if (handleCharacterInput(c))
        return true;
    return false;
}

void NvSampleApp::initRenderLoopObjects() {
    mTestModeTimer = createStopWatch();
    mTestModeFrames = -TESTMODE_WARMUP_FRAMES;
    mTotalTime = -1e6f; // don't exit during startup

    /*mFramerate = new NvFramerateCounter(this);

    mFrameTimer->start();*/

    mSumDrawTime = 0.0f;
    mDrawTimeFrames = 0;
    mDrawRate = 0.0f;
    mDrawTime = createStopWatch();
}

void NvSampleApp::shutdownRenderLoopObjects() {
    if (mHasInitializedRendering) {
        baseShutdownRendering();
        mHasInitializedRendering = false;
    }
}

void NvSampleApp::renderLoopRenderFrame() {
    mFrameTimer->stop();

    if (mTestMode) {
        // Simulate 60fps
        mFrameDelta = 1.0f / 60.0f;

        // just an estimate
        mTotalTime += mFrameTimer->getTime();
    } else {
        mFrameDelta = mFrameTimer->getTime();
        // just an estimate
        mTotalTime += mFrameDelta;
    }

	mFrameTimer->reset();
	
	if (m_width == 0 || m_height == 0) {
		NvThreadManager* thread = getThreadManagerInstance();

		if (thread)
			thread->sleepThread(200);

		return;
	}
	
	// initialization may cause the app to want to exit
    if (!isExiting()) {
        if (mEventTickTimer->getTime()>=0.05f) {
            mEventTickTimer->start(); // reset and continue...
        }

        mDrawTime->start();

		getAppContext()->beginFrame();

		getAppContext()->beginScene();
		baseDraw();

        if (mTestMode && (mTestRepeatFrames > 1)) {
            // repeat frame so that we can simulate a heavier workload
            for (int i = 1; i < mTestRepeatFrames; i++) {
                baseUpdate();
				baseDraw();
            }
        }

		getAppContext()->endScene();

		if (mTestMode && mUseFBOPair) {
			// Check if the app bound FBO 0 in FBO mode
			if (getAppContext()->isRenderingToMainScreen())
				m_testModeIssues |= TEST_MODE_FBO_ISSUE;
		}

		getAppContext()->endFrame();

        mDrawTime->stop();
        mSumDrawTime += mDrawTime->getTime();
        mDrawTime->reset();

        mDrawTimeFrames++;
        if (mDrawTimeFrames > 10) {
            mDrawRate = mDrawTimeFrames / mSumDrawTime;
            mDrawTimeFrames = 0;
            mSumDrawTime = 0.0f;
        }

		if (mLogFPS) {
			// wall time - not (possibly) simulated time
			mTimeSinceFPSLog += mFrameTimer->getTime();

			if (mTimeSinceFPSLog > 1.0f) {
				//LOGI("fps: %.2f", mFramerate->getMeanFramerate());
				mTimeSinceFPSLog = 0.0f;
			}
        }
    }

    if (mTestMode) {
        mTestModeFrames++;
        // if we've come to the end of the warm-up, start timing
        if (mTestModeFrames == 0) {
            mTotalTime = 0.0f;
            mTestModeTimer->start();
        }

        if (mTotalTime > mTestDuration) {
            mTestModeTimer->stop();
            double frameRate = mTestModeFrames / mTestModeTimer->getTime();
            logTestResults((float)frameRate, mTestModeFrames);
            exit(0);
        }
    }
}

bool NvSampleApp::haltRenderingThread() {
    // DO NOT test whether we WANT threading - the app may have just requested
    // threaded rendering to be disabled.
    // If threaded:
    // 1) Signal the rendering thread to exit
    if (mThread) {
        mRenderSync->set();
        mThread->signalQuit();
        // 2) Wait for the thread to complete (it will unbind the context), if it is running
        if (mThread->waitForQuit()) {
            // 3) Bind the context (unless it is lost?)
            getAppContext()->bindContext();
        }
        NV_DELETE_AND_RESET(mThread);
    }

    return true;
}

void* NvSampleApp::renderThreadThunk(void* thiz) {
    ((NvSampleApp*)thiz)->renderThreadFunc();
    return NULL;
}

void NvSampleApp::renderThreadFunc() {
    getAppContext()->prepThreadForRender();

    getAppContext()->bindContext();

    nvidia::shdfnd::memoryBarrier();
    mMainSync->set();

    while (mThread && !mThread->quitIsSignalled()) {
        renderLoopRenderFrame();

        // if we are not in full-bore rendering mode, wait to be triggered
        if (getPlatformContext()->getRedrawMode() != NvRedrawMode::UNBOUNDED) {
            mRenderSync->wait();
            mRenderSync->reset();
        }
    }

    getAppContext()->unbindContext();
    mRenderThreadRunning = false;
}

bool NvSampleApp::conditionalLaunchRenderingThread() {
    if (mUseRenderThread) {
        if (!mRenderThreadRunning) {
            // If threaded and the render thread is not running:
            // 1) Unbind the context
            getAppContext()->unbindContext();
            // 2) Call the thread launch function (which will bind the context)
            mRenderThreadRunning = true;
            mThread = NV_NEW(nvidia::shdfnd::Thread)(renderThreadThunk, this);

            // 3) WAIT for the rendering thread to bind or fail
            mMainSync->wait();
            mMainSync->reset();
        }

        // In any of the "triggered" modes, trigger the rendering thread loop
        if (getPlatformContext()->getRedrawMode() != NvRedrawMode::UNBOUNDED) {
            mRenderSync->set();
        }
        return true;
    } else {
        haltRenderingThread();

        // return false if we are not running in threaded mode or
        // _CANNOT_ support threading
        return false;
    }
}


void NvSampleApp::mainThreadRenderStep() 
{    

    // If we're ready to render (i.e. the GL is ready and we're focused), then go ahead
    //if (ctx->shouldRender()) 
    {
        // If we've not (re-)initialized the resources, do it
        if (!mHasInitializedRendering /*&& !isExiting()*/) 
        {
			mHasInitializedRendering = baseInitRendering();
        } 
      
        // TODO: Query size.
        m_width = 1920;
        m_height = 1080;
        renderLoopRenderFrame();
    }
}

void NvSampleApp::requestThreadedRendering(bool threaded) {
    mUseRenderThread = threaded;
}

bool NvSampleApp::isRenderThreadRunning() {
    return mRenderThreadRunning;
}

void NvSampleApp::mainLoop() {

    if (mTestMode) {
        writeLogFile(mTestName, false, "*** Starting Test\n");
    }

    mHasInitializedRendering = false;

    initRenderLoopObjects();

    // TBD - WAR for Android lifecycle change; this will be reorganized in the next release
#ifdef ANDROID
    while (getPlatformContext()->isAppRunning()) {
#else
    while (/*getPlatformContext()->isAppRunning() && !isExiting()*/1) {
#endif
        //getPlatformContext()->pollEvents(isAppInputHandlingEnabled() ? this : NULL);

        baseUpdate();

        mainThreadRenderStep();
    }

    //haltRenderingThread();

    //shutdownRenderLoopObjects();

    // mainloop exiting, clean up things created in mainloop lifespan.
}

void NvSampleApp::errorExit(const char* errorString) {
    if (mTestMode) {
        writeLogFile(mTestName, true, "Fatal Error from app\n");
        writeLogFile(mTestName, true, errorString);
        appRequestExit();
    } else {
        // we set the flag here manually.  The exit will not happen until
        // the user closes the dialog.  But we want to act as if we are
        // already exiting (which we are), so we do not render
        m_requestedExit = true;
        showDialog("Fatal Error", errorString, true);
    }
}

bool NvSampleApp::getRequestedWindowSize(int32_t& width, int32_t& height) {
    bool changed = false;
    if (m_desiredWidth != 0) {
        width = m_desiredWidth;
        changed = true;
    }

    if (m_desiredHeight != 0) {
        height = m_desiredHeight;
        changed = true;
    }

    return changed;
}

void NvSampleApp::baseShutdownRendering(void) {
    platformShutdownRendering();
    shutdownRendering();
}

void NvSampleApp::logTestResults(float frameRate, int32_t frames) {
    LOGI("Test Frame Rate = %lf (frames = %d) (%d x %d)\n", frameRate, frames, m_width, m_height);
    writeLogFile(mTestName, true, "\n%s %lf fps (%d frames) (%d x %d)\n", mTestName.c_str(), 
		frameRate, frames, m_width, m_height);

    if (m_testModeIssues != TEST_MODE_ISSUE_NONE) {
        writeLogFile(mTestName, true, "\nWARNING - there were potential test mode anomalies\n");

        if (m_testModeIssues & TEST_MODE_FBO_ISSUE) {
            writeLogFile(mTestName, true, "\tThe application appears to have explicitly bound the onscreen framebuffer\n"
                "\tSince the test was being run in offscreen rendering mode, this could invalidate results\n"
                "\tThe application should be checked for glBindFramebuffer of 0\n\n");
        }
    }
    platformLogTestResults(frameRate, frames);

    int32_t w = 1, h = 1;

    if (!getAppContext()->readFramebufferRGBX32(NULL, w, h)) {
        writeLogFile(mTestName, true, "\tThe application appears to have written to the onscreen framebuffer\n"
            "\tSince the test was being run in offscreen rendering mode, this could invalidate results\n"
            "\tThe application should be checked for glBindFramebuffer of 0\n\n");
    }

    uint8_t* data = new uint8_t[4 * m_width * m_height];

    getAppContext()->readFramebufferRGBX32(data, w, h);

    writeScreenShot(m_width, m_height, data, mTestName);
    writeLogFile(mTestName, true, "Test Complete!");

    delete[] data;
}
