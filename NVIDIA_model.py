def nvidia_model():

  model = Sequential()

  model.add(Conv2D(24, 5, strides=(2, 2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Conv2D(36, 5, strides=(2, 2), activation='elu'))
  model.add(Conv2D(48, 5, strides=(2, 2), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='elu'))
  model.add(Dropout(0.5))

  model.add(Flatten())

  model.add(Dense(100, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation='elu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='elu'))
  model.add(Dropout(0.5))

  model.add(Dense(1))
  model.compile(Adam(lr = 0.001), loss='mse')

  return model
  
model = nvidia_model()
print(model.summary())

history = model.fit(X_train, y_train, epochs=30, validation_data=(X_valid, y_valid), batch_size=100, verbose=1, shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epochs')
