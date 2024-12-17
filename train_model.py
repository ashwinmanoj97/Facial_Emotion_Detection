from data_loader import load_data
from model import create_model

train_dir = 'data/train/'
test_dir = 'data/test/'

# Load training and test data
train_data, test_data = load_data(train_dir, test_dir)

# Create and compile the model
model = create_model(input_shape=(48, 48, 1), num_classes=7)

# Train the model
model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=10,
    validation_data=test_data,
    validation_steps=test_data.samples // test_data.batch_size
)

# Save the trained model
model.save('models/emotion_model_final.h5')
print("Model saved as 'emotion_model_final.h5'")
