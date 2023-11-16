for i in range(1, epochs + 1):

    def closure():
        # Zero out the gradients
        optimizer.zero_grad()

        # Compute target features
        target_features = model_activations(target, model)

        # Compute content loss
        content_loss = torch.mean((content_features['conv4_2'] - target_features['conv4_2'])**2)

        # Dynamically adjust layer weights based on current target features
        style_wt_meas = adjust_layer_weights(style_features, content_features)

        # Compute style loss
        style_loss = 0
        for layer in style_wt_meas:
            style_gram = style_grams[layer]
            target_gram = target_features[layer]
            _, d, w, h = target_gram.shape
            target_gram = gram_matrix(target_gram)

            style_loss += (style_wt_meas[layer] * torch.mean((target_gram - style_gram)**2)) / (d * w * h)

        # Compute total variation loss
        tv_loss = total_variation_loss(target)

        # Combine losses
        total_loss = content_wt * content_loss + style_wt * style_loss + tv_loss_weight * tv_loss

        # Perform backward pass
        total_loss.backward()

        # Logging
        if i % 10 == 0:
            print("epoch ", i, " ", total_loss.item())

        return total_loss

    # Step through the optimizer
    optimizer.step(closure)

    # Checkpoint and display the image at intervals
    if i % print_after == 0:
        plt.imshow(imcnvt(target), label="Epoch " + str(i))
        plt.show()
        plt.imsave(str(i) + '.jpg', imcnvt(target), format='jpg')
