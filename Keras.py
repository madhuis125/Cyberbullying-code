import keras_ocr 
pipeline = keras_ocr.pipeline.Pipeline()
images = [
  keras_ocr.tools.read(dir) for dir in [
      'C:\\Users\\Pramod M\\Documents\\image3.jpg',        
      
  ]
]

prediction_groups = pipeline.recognize(images)
predicted_image_1 = prediction_groups[0]
for text, box in predicted_image_1:
    print(text)  
predicted_image_2 = prediction_groups[1]
for text, box in predicted_image_2:
    print(text)