#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "cnn.h"

// Função auxiliar para inicializar pesos aleatoriamente
void initialize_random_weights(CNN *cnn) {
    // Inicializa pesos da camada convolucional
    for (int f = 0; f < cnn->conv_layer->num_filters; ++f) {
        for (int d = 0; d < cnn->conv_layer->filters[f]->depth; ++d) {
            for (int r = 0; r < cnn->conv_layer->filters[f]->rows; ++r) {
                for (int c = 0; c < cnn->conv_layer->filters[f]->cols; ++c) {
                    setPixel(cnn->conv_layer->filters[f], d, r, c, ((scalar_t)rand() / RAND_MAX - 0.5) * 0.1);
                }
            }
        }
        setPixel(cnn->conv_layer->biases[f], 0, 0, 0, ((scalar_t)rand() / RAND_MAX - 0.5) * 0.1);
    }
    // Inicializa pesos da camada FC
    for (int i = 0; i < cnn->fc_layer->input_size * cnn->fc_layer->output_size; ++i) {
        cnn->fc_layer->weights[i] = ((scalar_t)rand() / RAND_MAX - 0.5) * 0.1;
    }
    for (int i = 0; i < cnn->fc_layer->output_size; ++i) {
        cnn->fc_layer->biases[i] = ((scalar_t)rand() / RAND_MAX - 0.5) * 0.1;
    }
}

int main() {
    srand(time(NULL)); // Inicializa o gerador de números aleatórios

    printf("\n--- Testando Forward Propagation da CNN Completa ---\n");

    // Definir dimensões da imagem de entrada e número de classes
    int input_depth = 1; // Imagem em tons de cinza
    int input_rows = 28; // Exemplo: imagem 28x28 (como MNIST)
    int input_cols = 28;
    int num_classes = 10; // Exemplo: 10 dígitos (0-9) para reconhecimento de caracteres

    // Criar a CNN
    CNN *my_cnn = createCNN(input_depth, input_rows, input_cols, num_classes);

    // Inicializar pesos aleatoriamente
    initialize_random_weights(my_cnn);

    // Criar uma imagem de entrada de exemplo (valores aleatórios)
    Image *test_image = createImage(input_depth, input_rows, input_cols);
    for (int i = 0; i < input_depth * input_rows * input_cols; ++i) {
        test_image->data[i] = (scalar_t)rand() / RAND_MAX;
    }

    printf("Processando imagem de entrada (%dx%dx%d)...\n", input_depth, input_rows, input_cols);

    // Executar o forward pass da CNN
    scalar_t *predictions = cnnForward(my_cnn, test_image);

    printf("Previsões da CNN (probabilidades):\n");
    for (int i = 0; i < num_classes; ++i) {
        printf("Classe %d: %.4f\n", i, predictions[i]);
    }

    // Encontrar a classe com maior probabilidade
    int predicted_class = 0;
    scalar_t max_prob = predictions[0];
    for (int i = 1; i < num_classes; ++i) {
        if (predictions[i] > max_prob) {
            max_prob = predictions[i];
            predicted_class = i;
        }
    }
    printf("Classe Prevista: %d (com probabilidade %.4f)\n", predicted_class, max_prob);

    // Liberar memória
    free(predictions);
    freeImage(test_image);
    freeCNN(my_cnn);

    return 0;
}

// Função de perda: Entropia Cruzada Categórica
scalar_t crossEntropyLoss(scalar_t *predictions, int true_label_idx, int num_classes) {
    // Para evitar log(0), que é indefinido, adicionamos um pequeno epsilon
    scalar_t epsilon = 1e-9;
    scalar_t true_prob = predictions[true_label_idx];
    return -log(true_prob + epsilon);
}


