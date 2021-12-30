weight = list(model.children())[1].weight.cpu().data
print(weight.shape)
fig = plt.figure(figsize=(28, 28))

for i in range(weight.shape[0]):
    image = weight[i].reshape((28, 28))

    # Adds a subplot at the 1st position
    fig.add_subplot(2, 5, i + 1)  # rows, colums, position

    # showing image
    plt.imshow(image, cmap="gray")
    plt.axis('off')
    plt.title(f"Number: {i}")

    model = nn.Sequential(
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        nn.Linear(28 * 28 * 1, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        # No need to include softmax, as this is already combined in the loss function
    )