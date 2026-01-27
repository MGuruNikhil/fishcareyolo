# Image Preprocessing TODO

## Overview

The TFLite inference pipeline is complete and functional, but the **image preprocessing step currently uses placeholder data** instead of actual image pixels. This document outlines the issue and potential solutions.

## Current Status

### What Works
- ✅ TFLite model loading and caching
- ✅ Model inference execution via `react-native-fast-tflite`
- ✅ YOLO output parsing
- ✅ NMS (Non-Maximum Suppression)
- ✅ Confidence filtering and sorting
- ✅ Detection object creation
- ✅ ModelContext integration

### What Needs Work
- ❌ Image decoding to RGB pixel data

## The Problem

**Location**: `app/lib/model/inference.ts` - `preprocessImage()` function (lines 264-298)

**Issue**: The function currently:
1. Resizes image to 640x640 using `expo-image-manipulator`
2. Creates a placeholder `Uint8Array` filled with gray values (128, 128, 128)
3. Returns this placeholder instead of actual image pixels

**Why**: Converting JPEG/PNG images to raw RGB pixel data is non-trivial in React Native without specialized libraries or native code.

**Impact**: 
- Inference runs successfully but produces meaningless results
- The model receives uniform gray pixels instead of actual image content
- Rest of the app (UI, navigation, storage) can still be built and tested

## Solution Options

### Option 1: VisionCamera Integration (Recommended for Real-Time)

**Best for**: Real-time camera detection, live preview with bounding boxes

**Dependencies**:
```bash
npm install react-native-vision-camera
npm install vision-camera-resize-plugin
```

**Implementation**:
- Use VisionCamera Frame Processors
- `vision-camera-resize-plugin` provides `resize()` function
- Returns properly formatted RGB tensor directly from camera frames

**Example**:
```typescript
import { useResizePlugin } from 'vision-camera-resize-plugin'

const { resize } = useResizePlugin()

const frameProcessor = useFrameProcessor((frame) => {
  'worklet'
  const resized = resize(frame, {
    scale: { width: 640, height: 640 },
    pixelFormat: 'rgb',
    dataType: 'uint8',
  })
  // resized is now a Uint8Array ready for TFLite
}, [])
```

**Pros**:
- Clean integration with VisionCamera
- Optimized for real-time performance
- Well-maintained plugin

**Cons**:
- Requires VisionCamera setup
- Only works with camera frames (not gallery images)
- Adds dependency

### Option 2: Native Module for Image Decoding (Recommended for Gallery Images)

**Best for**: Processing gallery images, offline analysis

**Implementation**:
Create native modules for iOS and Android that decode images to RGB:

**iOS (Swift/Objective-C)**:
```swift
// Use UIImage and CoreGraphics
func decodeImageToRGB(imageUri: String) -> [UInt8] {
    guard let image = UIImage(contentsOfFile: imageUri) else { return [] }
    guard let cgImage = image.cgImage else { return [] }
    
    let width = 640
    let height = 640
    let bytesPerPixel = 3
    let bytesPerRow = width * bytesPerPixel
    
    var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)
    
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let context = CGContext(
        data: &pixelData,
        width: width,
        height: height,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: colorSpace,
        bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
    )
    
    context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    
    return pixelData
}
```

**Android (Kotlin)**:
```kotlin
// Use Bitmap
fun decodeImageToRGB(imageUri: String): ByteArray {
    val options = BitmapFactory.Options().apply {
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    
    val bitmap = BitmapFactory.decodeFile(imageUri, options)
    val scaledBitmap = Bitmap.createScaledBitmap(bitmap, 640, 640, true)
    
    val pixels = IntArray(640 * 640)
    scaledBitmap.getPixels(pixels, 0, 640, 0, 0, 640, 640)
    
    val rgb = ByteArray(640 * 640 * 3)
    var idx = 0
    for (pixel in pixels) {
        rgb[idx++] = ((pixel shr 16) and 0xFF).toByte() // R
        rgb[idx++] = ((pixel shr 8) and 0xFF).toByte()  // G
        rgb[idx++] = (pixel and 0xFF).toByte()          // B
    }
    
    return rgb
}
```

**Pros**:
- Full control over image decoding
- Works with any image source (camera, gallery, file system)
- Optimal performance

**Cons**:
- Requires native code maintenance
- More complex setup
- Platform-specific implementations

### Option 3: expo-gl or react-native-skia

**Best for**: Projects already using GL or Skia

**Dependencies**:
```bash
# Option A: expo-gl
npm install expo-gl

# Option B: react-native-skia
npm install @shopify/react-native-skia
```

**Implementation (expo-gl)**:
```typescript
import { GLView } from 'expo-gl'
import * as FileSystem from 'expo-file-system'

async function decodeImageToRGB(imageUri: string): Promise<Uint8Array> {
  // Load image into GL texture
  const gl = await GLView.createContextAsync()
  const texture = await gl.createTextureFromAssetAsync({ uri: imageUri })
  
  // Read pixels from texture
  const pixels = new Uint8Array(640 * 640 * 4) // RGBA
  gl.readPixels(0, 0, 640, 640, gl.RGBA, gl.UNSIGNED_BYTE, pixels)
  
  // Convert RGBA to RGB
  const rgb = new Uint8Array(640 * 640 * 3)
  for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
    rgb[j] = pixels[i]     // R
    rgb[j+1] = pixels[i+1] // G
    rgb[j+2] = pixels[i+2] // B
  }
  
  return rgb
}
```

**Pros**:
- Pure JavaScript solution
- Cross-platform
- Leverages existing libraries

**Cons**:
- Adds heavyweight dependency for simple task
- GL context overhead
- May not work in all environments (background processing)

### Option 4: Third-Party Library

**Best for**: Quick solution if library exists and is maintained

**Potential Libraries**:
- `react-native-image-to-tensor` (check if exists)
- `@tensorflow/tfjs-react-native` image utilities
- `react-native-fast-image` with pixel access

**Implementation**:
```typescript
// Hypothetical - verify library availability
import { imageToTensor } from 'react-native-image-to-tensor'

const tensor = await imageToTensor(imageUri, {
  width: 640,
  height: 640,
  format: 'rgb',
})
```

**Pros**:
- Quickest implementation if library is good
- Community maintained

**Cons**:
- May not exist or be maintained
- Dependency risk
- Less control

## Recommended Approach

**For this project**: Implement **Option 2 (Native Module)** because:
1. ✅ Works with both camera and gallery images
2. ✅ Optimal performance for production
3. ✅ Full control over image processing
4. ✅ No additional dependencies beyond what we have
5. ✅ Platform-specific optimizations possible

**Alternative**: If VisionCamera is already planned for other features, use **Option 1** for camera images and **Option 2** for gallery images.

## Implementation Steps (When Ready)

1. **Create Native Module Structure**
   ```
   app/
     android/
       src/main/java/com/mina/ImageDecoderModule.kt
     ios/
       ImageDecoderModule.swift
       ImageDecoderModule.m (bridge)
   ```

2. **Implement iOS Decoder**
   - Use UIImage + CoreGraphics
   - Decode to RGB888 format
   - Return as NSData/Data

3. **Implement Android Decoder**
   - Use BitmapFactory + Bitmap
   - Decode to RGB888 format
   - Return as byte array

4. **Create TypeScript Bridge**
   ```typescript
   // app/lib/model/image-decoder.ts
   import { NativeModules } from 'react-native'
   
   const { ImageDecoder } = NativeModules
   
   export async function decodeImageToRGB(
     imageUri: string
   ): Promise<Uint8Array> {
     const rgb = await ImageDecoder.decodeToRGB(imageUri, 640, 640)
     return new Uint8Array(rgb)
   }
   ```

5. **Update preprocessImage() in inference.ts**
   - Replace placeholder logic with native decoder call
   - Remove warning message
   - Add proper error handling

6. **Test on Both Platforms**
   - iOS device
   - Android device
   - Various image formats (JPEG, PNG)
   - Various image sizes

## Current Workaround

Until image preprocessing is implemented, the app will:
- ✅ Run inference successfully
- ✅ Return detection results (but likely empty/random)
- ✅ Allow full UI/UX testing
- ⚠️ Not provide meaningful disease detection

**The rest of the app can be built and tested without blocking on this issue.**

## Priority

**Priority**: Medium (not blocking for Task 8 - Results Screen)

**Recommended Timeline**: 
- Complete Task 8 (Results Screen) first
- Build full app flow and UI
- Then implement proper image preprocessing
- Test end-to-end with real results

## References

- [react-native-fast-tflite documentation](https://github.com/mrousavy/react-native-fast-tflite)
- [vision-camera-resize-plugin](https://github.com/mrousavy/vision-camera-resize-plugin)
- [iOS CoreGraphics Image Processing](https://developer.apple.com/documentation/coregraphics)
- [Android Bitmap Documentation](https://developer.android.com/reference/android/graphics/Bitmap)
