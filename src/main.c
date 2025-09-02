#include <stdio.h>
#include <stdlib.h>
#include "cnn.h"

int main() {
    printf("\n--- Testando Camada Convolucional ---\n");
    // Criar uma imagem de entrada simples (1 canal, 5x5)
    Image *input_conv = createImage(1, 5, 5);
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 5; ++c) {
            setPixel(input_conv, 0, r, c, (scalar_t)(r * 5 + c + 1));
        }
    }
    printf("Input para Convolução:\n");
    for (int r = 0; r < input_conv->rows; ++r) {
        for (int c = 0; c < input_conv->cols; ++c) {
            printf("%.1f ", getPixel(input_conv, 0, r, c));
        }
        printf("\n");
    }

    // Criar uma camada convolucional (1 filtro, kernel 3x3, stride 1, padding 0)
    ConvLayer *conv_layer = createConvLayer(1, 3, 1, 0, 1);
    // Definir um filtro simples para teste: todos os valores 1.0
    for (int kr = 0; kr < 3; ++kr) {
        for (int kc = 0; kc < 3; ++kc) {
            setPixel(conv_layer->filters[0], 0, kr, kc, 1.0);
        }
    }
    setPixel(conv_layer->biases[0], 0, 0, 0, 0.0); // Sem bias por enquanto

    Image *output_conv = convForward(conv_layer, input_conv);
    printf("Output da Convolução (Feature Map):\n");
    for (int r = 0; r < output_conv->rows; ++r) {
        for (int c = 0; c < output_conv->cols; ++c) {
            printf("%.1f ", getPixel(output_conv, 0, r, c));
        }
        printf("\n");
    }

    printf("\n--- Testando Camada ReLU ---\n");
    // Criar uma imagem de entrada com valores negativos para ReLU
    Image *input_relu = createImage(1, 3, 3);
    setPixel(input_relu, 0, 0, 0, -2.0); setPixel(input_relu, 0, 0, 1, 1.0); setPixel(input_relu, 0, 0, 2, -3.0);
    setPixel(input_relu, 0, 1, 0, 4.0); setPixel(input_relu, 0, 1, 1, -5.0); setPixel(input_relu, 0, 1, 2, 6.0);
    setPixel(input_relu, 0, 2, 0, -7.0); setPixel(input_relu, 0, 2, 1, 8.0); setPixel(input_relu, 0, 2, 2, -9.0);
    printf("Input para ReLU:\n");
    for (int r = 0; r < input_relu->rows; ++r) {
        for (int c = 0; c < input_relu->cols; ++c) {
            printf("%.1f ", getPixel(input_relu, 0, r, c));
        }
        printf("\n");
    }

    Image *output_relu = reluForward(input_relu);
    printf("Output da ReLU:\n");
    for (int r = 0; r < output_relu->rows; ++r) {
        for (int c = 0; c < output_relu->cols; ++c) {
            printf("%.1f ", getPixel(output_relu, 0, r, c));
        }
        printf("\n");
    }

    printf("\n--- Testando Camada de Pooling (Max Pooling) ---\n");
    // Criar uma imagem de entrada para Pooling (1 canal, 4x4)
    Image *input_pool = createImage(1, 4, 4);
    setPixel(input_pool, 0, 0, 0, 1.0); setPixel(input_pool, 0, 0, 1, 2.0); setPixel(input_pool, 0, 0, 2, 3.0); setPixel(input_pool, 0, 0, 3, 4.0);
    setPixel(input_pool, 0, 1, 0, 5.0); setPixel(input_pool, 0, 1, 1, 6.0); setPixel(input_pool, 0, 1, 2, 7.0); setPixel(input_pool, 0, 1, 3, 8.0);
    setPixel(input_pool, 0, 2, 0, 9.0); setPixel(input_pool, 0, 2, 1, 10.0); setPixel(input_pool, 0, 2, 2, 11.0); setPixel(input_pool, 0, 2, 3, 12.0);
    setPixel(input_pool, 0, 3, 0, 13.0); setPixel(input_pool, 0, 3, 1, 14.0); setPixel(input_pool, 0, 3, 2, 15.0); setPixel(input_pool, 0, 3, 3, 16.0);
    printf("Input para Pooling:\n");
    for (int r = 0; r < input_pool->rows; ++r) {
        for (int c = 0; c < input_pool->cols; ++c) {
            printf("%.1f ", getPixel(input_pool, 0, r, c));
        }
        printf("\n");
    }

    // Criar uma camada de pooling (pool_size 2x2, stride 2)
    PoolLayer *pool_layer = createPoolLayer(2, 2);

    Image *output_pool = poolForward(pool_layer, input_pool);
    printf("Output do Pooling:\n");
    for (int r = 0; r < output_pool->rows; ++r) {
        for (int c = 0; c < output_pool->cols; ++c) {
            printf("%.1f ", getPixel(output_pool, 0, r, c));
        }
        printf("\n");
    }

    // Liberar memória
    freeImage(input_conv);
    freeImage(output_conv);
    freeConvLayer(conv_layer);

    freeImage(input_relu);
    freeImage(output_relu);

    freeImage(input_pool);
    freeImage(output_pool);
    freePoolLayer(pool_layer);

    printf("\n--- Testando Camada Totalmente Conectada (FC) e Softmax ---\n");

    // Simular a saída de uma camada anterior (e.g., após pooling e flatten)
    // Vamos criar uma imagem 1x2x2 (profundidade 1, 2x2) e achatá-la
    Image *simulated_input_image = createImage(1, 2, 2);
    setPixel(simulated_input_image, 0, 0, 0, 0.1);
    setPixel(simulated_input_image, 0, 0, 1, 0.5);
    setPixel(simulated_input_image, 0, 1, 0, 0.2);
    setPixel(simulated_input_image, 0, 1, 1, 0.8);

    scalar_t *flattened_input = flattenImage(simulated_input_image);
    int input_fc_size = simulated_input_image->depth * simulated_input_image->rows * simulated_input_image->cols;
    printf("Input achatado para FC (tamanho %d): ", input_fc_size);
    for (int i = 0; i < input_fc_size; ++i) {
        printf("%.2f ", flattened_input[i]);
    }
    printf("\n");

    // Criar uma camada FC (entrada 4, saída 3 - simulando 3 classes de saída)
    FCLayer *fc_layer = createFCLayer(input_fc_size, 3);

    // Definir pesos e vieses para teste (valores arbitrários)
    // Pesos (input_size x output_size) - exemplo para 4x3
    // Neuron 0 (output)
    fc_layer->weights[0 * 3 + 0] = 0.1; fc_layer->weights[1 * 3 + 0] = 0.2; fc_layer->weights[2 * 3 + 0] = 0.3; fc_layer->weights[3 * 3 + 0] = 0.4;
    // Neuron 1 (output)
    fc_layer->weights[0 * 3 + 1] = 0.5; fc_layer->weights[1 * 3 + 1] = 0.6; fc_layer->weights[2 * 3 + 1] = 0.7; fc_layer->weights[3 * 3 + 1] = 0.8;
    // Neuron 2 (output)
    fc_layer->weights[0 * 3 + 2] = 0.9; fc_layer->weights[1 * 3 + 2] = 1.0; fc_layer->weights[2 * 3 + 2] = 1.1; fc_layer->weights[3 * 3 + 2] = 1.2;

    // Vieses (output_size)
    fc_layer->biases[0] = 0.01;
    fc_layer->biases[1] = 0.02;
    fc_layer->biases[2] = 0.03;

    scalar_t *fc_output = fcForward(fc_layer, flattened_input);
    printf("Output da Camada FC (logits): ");
    for (int i = 0; i < fc_layer->output_size; ++i) {
        printf("%.2f ", fc_output[i]);
    }
    printf("\n");

    scalar_t *softmax_output = softmaxForward(fc_output, fc_layer->output_size);
    printf("Output da Softmax (probabilidades): ");
    for (int i = 0; i < fc_layer->output_size; ++i) {
        printf("%.4f ", softmax_output[i]);
    }
    printf("\n");

    // Liberar memória
    freeImage(simulated_input_image);
    free(flattened_input);
    free(fc_output);
    free(softmax_output);
    freeFCLayer(fc_layer);

    return 0;
}