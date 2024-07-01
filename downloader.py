from keras.applications import MobileNetV2

# Download MobileNetV2 model
mobile_net_v2_model = MobileNetV2(weights='imagenet', include_top=False)
mobile_net_v2_model.save('mobilenetv2_model.keras')
