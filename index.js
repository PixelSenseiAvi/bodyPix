import * as bodyPix from "@tensorflow-models/body-pix";
import * as partColorScales from "./color-scales";

const state = {
  video: null,
  stream: null,
  net: null,
  videoConstraints: {},
  changingCamera: false,
  changingArchitecture: false,
};

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

async function getVideoInputs() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
    console.log("enumerateDevices() not supported");
    return [];
  }

  const devices = await navigator.mediaDevices.enumerateDevices();

  const videoDevices = devices.filter((device) => device.kind === "videoinput");

  return videoDevices;
}

function stopExistingVideoCapture() {
  if (state.video && state.video.srcObject) {
    state.video.srcObject.getTracks().forEach((track) => {
      track.stop();
    });
    state.video.srcObject = null;
  }
}

async function getDeviceForLabel(cameraLabel) {
  const videoInputs = await getVideoInputs();

  for (let i = 0; i < videoInputs.length; i++) {
    const videoInput = videoInputs[i];
    if (videoInput.label === cameraLabel) {
      return videoInput.deviceId;
    }
  }

  return null;
}

function getFacingMode(cameraLabel) {
  if (!cameraLabel) {
    return "user";
  }
  if (cameraLabel.toLowerCase().includes("back")) {
    return "environment";
  } else {
    return "user";
  }
}

async function getConstraints(cameraLabel) {
  let deviceId, facingMode;

  if (cameraLabel) {
    deviceId = await getDeviceForLabel(cameraLabel);
    facingMode = isMobile() ? getFacingMode(cameraLabel) : null;
  }
  return { deviceId, facingMode };
}

async function setupCamera(cameraLabel) {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      "Browser API navigator.mediaDevices.getUserMedia not available"
    );
  }

  const videoElement = document.getElementById("video");

  stopExistingVideoCapture();

  const videoConstraints = await getConstraints(cameraLabel);

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: videoConstraints,
  });
  videoElement.srcObject = stream;

  return new Promise((resolve) => {
    videoElement.onloadedmetadata = () => {
      videoElement.width = videoElement.videoWidth;
      videoElement.height = videoElement.videoHeight;
      resolve(videoElement);
    };
  });
}

async function loadVideo(cameraLabel) {
  try {
    state.video = await setupCamera(cameraLabel);
  } catch (e) {
    let info = document.getElementById("info");
    info.textContent =
      "this browser does not support video capture," +
      "or this device does not have a camera";
    info.style.display = "block";
    throw e;
  }

  state.video.play();
}

const guiState = {
  estimate: "segmentation",
  camera: null,
  flipHorizontal: true,
  input: {
    mobileNetArchitecture: "1",
    outputStride: 16,
  },
  segmentation: {
    segmentationThreshold: 0.5,
    effect: "mask",
    maskBackground: true,
    opacity: 0.7,
    backgroundBlurAmount: 3,
    maskBlurAmount: 0,
    edgeBlurAmount: 3,
  },
  partMap: {
    colorScale: "rainbow",
    segmentationThreshold: 0.5,
    applyPixelation: false,
    opacity: 0.9,
  },
  showFps: false,
};

function segmentBodyInRealTime() {
  const canvas = document.getElementById("output");

  async function bodySegmentationFrame() {
    if (state.changingArchitecture || state.changingCamera) {
      setTimeout(bodySegmentationFrame, 1000);
      return;
    }

    const outputStride = +guiState.input.outputStride;
    const flipHorizontal = guiState.flipHorizontal;

    const personSegmentation = await state.net.estimatePersonSegmentation(
      state.video,
      outputStride,
      guiState.segmentation.segmentationThreshold
    );
    const mask = bodyPix.toMaskImageData(
      personSegmentation,
      guiState.segmentation.maskBackground
    );
    bodyPix.drawMask(
      canvas,
      state.video,
      mask,
      guiState.segmentation.opacity,
      guiState.segmentation.maskBlurAmount,
      flipHorizontal
    );

    requestAnimationFrame(bodySegmentationFrame);
  }
  bodySegmentationFrame();
}

function partEstimationInRealTime() {
  const canvas = document.getElementById("output");

  async function partSegmentationFrame() {
    if (state.changingArchitecture || state.changingCamera) {
      setTimeout(partSegmentationFrame, 1000);
      return;
    }

    // const img = document.getElementById("image");

    // const net = await bodyPix.load();
    // const partSegmentation = await net.segmentMultiPersonParts(img);
    console.log(state.net);
    debugger;
    const partSegmentation = await state.net.segmentMultiPersonParts(
      state.video,
      {
        flipHorizontal: false,
        internalResolution: "medium",
        segmentationThreshold: 0.7,
        maxDetections: 10,
        scoreThreshold: 0.2,
        nmsRadius: 20,
        minKeypointScore: 0.3,
        refineSteps: 10,
      }
    );

    const rainbow = [
      [110, 64, 170],
      [106, 72, 183],
      [100, 81, 196],
      [92, 91, 206],
      [84, 101, 214],
      [75, 113, 221],
      [66, 125, 224],
      [56, 138, 226],
      [48, 150, 224],
      [40, 163, 220],
      [33, 176, 214],
      [29, 188, 205],
      [26, 199, 194],
      [26, 210, 182],
      [28, 219, 169],
      [33, 227, 155],
      [41, 234, 141],
      [51, 240, 128],
      [64, 243, 116],
      [79, 246, 105],
      [96, 247, 97],
      [115, 246, 91],
      [134, 245, 88],
      [155, 243, 88],
    ];

    // the colored part image is an rgb image with a corresponding color from the rainbow colors for each part at each pixel.
    const coloredPartImage = bodyPix.toColoredPartMask(
      partSegmentation,
      rainbow
    );
    const opacity = 0.7;

    // draw the colored part image on top of the original image onto a canvas.  The colored part image will be drawn semi-transparent, with an opacity of 0.7, allowing for the original image to be visible under.
    bodyPix.drawMask(canvas, state.video, coloredPartImage, opacity);

    requestAnimationFrame(partSegmentationFrame);
  }
  partSegmentationFrame();
}

export async function bindPage() {
  state.net = await bodyPix.load({
    architecture: "ResNet50",
    outputStride: 32,
    quantBytes: 2,
  });
  document.getElementById("loading").style.display = "none";
  document.getElementById("main").style.display = "inline-block";

  await loadVideo();

  //   segmentBodyInRealTime();
  partEstimationInRealTime();
}

navigator.getUserMedia =
  navigator.getUserMedia ||
  navigator.webkitGetUserMedia ||
  navigator.mozGetUserMedia;

bindPage();
