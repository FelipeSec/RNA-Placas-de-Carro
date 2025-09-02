#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cnn.h"

// Funções auxiliares para manipulação de imagens/feature maps
Image* createImage(int depth, int rows, int cols) {
    Image *img = (Image*)malloc(sizeof(Image));
    if (!img) { fprintf(stderr, "Erro ao alocar Image\n"); exit(1); }
    img->depth = depth;
    img->rows = rows;
    img->cols = cols;
    img->data = (scalar_t*)calloc(depth * rows * cols, sizeof(scalar_t));
    if (!img->data) { fprintf(stderr, "Erro ao alocar dados da Image\n"); exit(1); }
    return img;
}

void freeImage(Image *img) {
    if (img) {
        free(img->data);
        free(img);
    }
}

// Função para obter o valor de um pixel (com verificação de limites)
scalar_t getPixel(Image *img, int d, int r, int c) {
    if (d < 0 || d >= img->depth || r < 0 || r >= img->rows || c < 0 || c >= img->cols) {
        // fprintf(stderr, "Aviso: Acesso fora dos limites da imagem em getPixel.\n");
        return 0.0; // Retorna 0 para padding ou acesso inválido
    }
    return img->data[d * img->rows * img->cols + r * img->cols + c];
}

// Função para definir o valor de um pixel
void setPixel(Image *img, int d, int r, int c, scalar_t value) {
    if (d < 0 || d >= img->depth || r < 0 || r >= img->rows || c < 0 || c >= img->cols) {
        fprintf(stderr, "Erro: Acesso fora dos limites da imagem em setPixel.\n");
        exit(1);
    }
    img->data[d * img->rows * img->cols + r * img->cols + c] = value;
}

// Funções para inicializar e liberar camadas
ConvLayer* createConvLayer(int num_filters, int kernel_size, int stride, int padding, int input_depth) {
    ConvLayer *layer = (ConvLayer*)malloc(sizeof(ConvLayer));
    if (!layer) { fprintf(stderr, "Erro ao alocar ConvLayer\n"); exit(1); }
    layer->num_filters = num_filters;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;

    // Alocar filtros e vieses
    layer->filters = (Image**)malloc(num_filters * sizeof(Image*));
    layer->biases = (Image**)malloc(num_filters * sizeof(Image*));
    if (!layer->filters || !layer->biases) { fprintf(stderr, "Erro ao alocar filtros/biases da ConvLayer\n"); exit(1); }

    for (int i = 0; i < num_filters; ++i) {
        // Cada filtro tem a mesma profundidade da entrada e tamanho kernel_size x kernel_size
        layer->filters[i] = createImage(input_depth, kernel_size, kernel_size);
        // Cada bias é um único valor, mas representamos como Image para consistência
        layer->biases[i] = createImage(1, 1, 1);
        // Inicializar pesos e vieses com valores aleatórios pequenos (ex: Glorot/Xavier ou He)
        // Por enquanto, vamos usar valores fixos para teste ou zeros
        // Para uma implementação real, use uma função de inicialização aleatória
    }
    return layer;
}

void freeConvLayer(ConvLayer *layer) {
    if (layer) {
        for (int i = 0; i < layer->num_filters; ++i) {
            freeImage(layer->filters[i]);
            freeImage(layer->biases[i]);
        }
        free(layer->filters);
        free(layer->biases);
        free(layer);
    }
}

PoolLayer* createPoolLayer(int pool_size, int stride) {
    PoolLayer *layer = (PoolLayer*)malloc(sizeof(PoolLayer));
    if (!layer) { fprintf(stderr, "Erro ao alocar PoolLayer\n"); exit(1); }
    layer->pool_size = pool_size;
    layer->stride = stride;
    return layer;
}

void freePoolLayer(PoolLayer *layer) {
    if (layer) {
        free(layer);
    }
}

// Funções de forward pass para as camadas

// Função de convolução
Image* convForward(ConvLayer *layer, Image *input) {
    // Calcular dimensões de saída
    int output_rows = (input->rows - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    int output_cols = (input->cols - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;

    Image *output = createImage(layer->num_filters, output_rows, output_cols);

    for (int f = 0; f < layer->num_filters; ++f) { // Para cada filtro
        Image *current_filter = layer->filters[f];
        scalar_t current_bias = getPixel(layer->biases[f], 0, 0, 0);

        for (int r_out = 0; r_out < output_rows; ++r_out) { // Para cada linha na saída
            for (int c_out = 0; c_out < output_cols; ++c_out) { // Para cada coluna na saída
                scalar_t sum = 0.0;
                // Calcular a posição inicial na entrada (com padding)
                int r_start = r_out * layer->stride - layer->padding;
                int c_start = c_out * layer->stride - layer->padding;

                for (int d_in = 0; d_in < input->depth; ++d_in) { // Para cada canal de entrada
                    for (int kr = 0; kr < layer->kernel_size; ++kr) { // Para cada linha do kernel
                        for (int kc = 0; kc < layer->kernel_size; ++kc) { // Para cada coluna do kernel
                            scalar_t input_val = getPixel(input, d_in, r_start + kr, c_start + kc);
                            scalar_t filter_val = getPixel(current_filter, d_in, kr, kc);
                            sum += input_val * filter_val;
                        }
                    }
                }
                setPixel(output, f, r_out, c_out, sum + current_bias);
            }
        }
    }
    return output;
}

// Função de ativação ReLU
Image* reluForward(Image *input) {
    Image *output = createImage(input->depth, input->rows, input->cols);
    for (int i = 0; i < input->depth * input->rows * input->cols; ++i) {
        output->data[i] = fmax(0.0, input->data[i]);
    }
    return output;
}

// Função de pooling (Max Pooling)
Image* poolForward(PoolLayer *layer, Image *input) {
    // Calcular dimensões de saída
    int output_rows = (input->rows - layer->pool_size) / layer->stride + 1;
    int output_cols = (input->cols - layer->pool_size) / layer->stride + 1;

    Image *output = createImage(input->depth, output_rows, output_cols);

    for (int d = 0; d < input->depth; ++d) { // Para cada canal
        for (int r_out = 0; r_out < output_rows; ++r_out) { // Para cada linha na saída
            for (int c_out = 0; c_out < output_cols; ++c_out) { // Para cada coluna na saída
                scalar_t max_val = -INFINITY; // Inicializa com valor muito pequeno
                // Calcular a posição inicial na entrada
                int r_start = r_out * layer->stride;
                int c_start = c_out * layer->stride;

                for (int pr = 0; pr < layer->pool_size; ++pr) { // Para cada linha da janela de pooling
                    for (int pc = 0; pc < layer->pool_size; ++pc) { // Para cada coluna da janela de pooling
                        scalar_t pixel_val = getPixel(input, d, r_start + pr, c_start + pc);
                        if (pixel_val > max_val) {
                            max_val = pixel_val;
                        }
                    }
                }
                setPixel(output, d, r_out, c_out, max_val);
            }
        }
    }
    return output;
}

// Função para "achatar" uma Image em um vetor 1D
scalar_t* flattenImage(Image *img) {
    scalar_t *flat_vector = (scalar_t*)malloc(img->depth * img->rows * img->cols * sizeof(scalar_t));
    if (!flat_vector) { fprintf(stderr, "Erro ao alocar vetor achatado\n"); exit(1); }
    for (int i = 0; i < img->depth * img->rows * img->cols; ++i) {
        flat_vector[i] = img->data[i];
    }
    return flat_vector;
}

// Funções para inicializar e liberar a camada FC
FCLayer* createFCLayer(int input_size, int output_size) {
    FCLayer *layer = (FCLayer*)malloc(sizeof(FCLayer));
    if (!layer) { fprintf(stderr, "Erro ao alocar FCLayer\n"); exit(1); }
    layer->input_size = input_size;
    layer->output_size = output_size;

    // Alocar pesos e vieses
    layer->weights = (scalar_t*)calloc(input_size * output_size, sizeof(scalar_t));
    layer->biases = (scalar_t*)calloc(output_size, sizeof(scalar_t));
    if (!layer->weights || !layer->biases) { fprintf(stderr, "Erro ao alocar pesos/biases da FCLayer\n"); exit(1); }

    // Inicializar pesos com valores aleatórios pequenos (e.g., Glorot/Xavier ou He)
    // Para este guia, vamos usar zeros para simplificar, mas em um projeto real, use inicialização adequada.
    // Exemplo de inicialização simples (não ideal para redes profundas):
    // for (int i = 0; i < input_size * output_size; ++i) layer->weights[i] = ((scalar_t)rand() / RAND_MAX - 0.5) * 0.01;
    // for (int i = 0; i < output_size; ++i) layer->biases[i] = 0.0;

    return layer;
}

void freeFCLayer(FCLayer *layer) {
    if (layer) {
        free(layer->weights);
        free(layer->biases);
        free(layer);
    }
}

// Função de forward pass para a camada FC
scalar_t* fcForward(FCLayer *layer, scalar_t *input_vector) {
    scalar_t *output_vector = (scalar_t*)calloc(layer->output_size, sizeof(scalar_t));
    if (!output_vector) { fprintf(stderr, "Erro ao alocar output_vector da FC\n"); exit(1); }

    for (int j = 0; j < layer->output_size; ++j) { // Para cada neurônio de saída
        scalar_t sum = 0.0;
        for (int i = 0; i < layer->input_size; ++i) { // Somar as entradas ponderadas
            sum += input_vector[i] * layer->weights[i * layer->output_size + j]; // weights[input_idx][output_idx]
        }
        output_vector[j] = sum + layer->biases[j];
    }
    return output_vector;
}

// Função de ativação Softmax
scalar_t* softmaxForward(scalar_t *input_vector, int size) {
    scalar_t *output_vector = (scalar_t*)calloc(size, sizeof(scalar_t));
    if (!output_vector) { fprintf(stderr, "Erro ao alocar output_vector da Softmax\n"); exit(1); }

    scalar_t sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        output_vector[i] = exp(input_vector[i]); // Calcula e^x para cada elemento
        sum_exp += output_vector[i];
    }

    for (int i = 0; i < size; ++i) {
        output_vector[i] /= sum_exp; // Normaliza para obter probabilidades
    }
    return output_vector;
}
// Funções para inicializar e liberar a CNN
CNN* createCNN(int input_depth, int input_rows, int input_cols, int num_classes) {
    CNN *cnn = (CNN*)malloc(sizeof(CNN));
    if (!cnn) { fprintf(stderr, "Erro ao alocar CNN\n"); exit(1); }

    // Parâmetros para a primeira camada convolucional
    int conv_num_filters = 8; // Exemplo: 8 filtros
    int conv_kernel_size = 3; // Exemplo: kernel 3x3
    int conv_stride = 1;
    int conv_padding = 0;
    cnn->conv_layer = createConvLayer(conv_num_filters, conv_kernel_size, conv_stride, conv_padding, input_depth);

    // Calcular dimensões de saída da camada convolucional
    int conv_output_rows = (input_rows - conv_kernel_size + 2 * conv_padding) / conv_stride + 1;
    int conv_output_cols = (input_cols - conv_kernel_size + 2 * conv_padding) / conv_stride + 1;

    // Parâmetros para a camada de pooling
    int pool_size = 2; // Exemplo: pool 2x2
    int pool_stride = 2;
    cnn->pool_layer = createPoolLayer(pool_size, pool_stride);

    // Calcular dimensões de saída da camada de pooling
    int pool_output_rows = (conv_output_rows - pool_size) / pool_stride + 1;
    int pool_output_cols = (conv_output_cols - pool_size) / pool_stride + 1;

    // Parâmetros para a camada totalmente conectada
    // A entrada da FC é o tamanho achatado da saída do pooling
    int fc_input_size = conv_num_filters * pool_output_rows * pool_output_cols;
    cnn->fc_layer = createFCLayer(fc_input_size, num_classes);

    return cnn;
}

void freeCNN(CNN* cnn) {
    if (cnn) {
        freeConvLayer(cnn->conv_layer);
        freePoolLayer(cnn->pool_layer);
        freeFCLayer(cnn->fc_layer);
        free(cnn);
    }
}

// Função de forward pass para a CNN completa
scalar_t* cnnForward(CNN* cnn, Image* input_image) {
    // 1. Camada Convolucional
    Image* conv_output = convForward(cnn->conv_layer, input_image);

    // 2. Camada ReLU (aplicada após a convolução)
    Image* relu_output = reluForward(conv_output);
    freeImage(conv_output); // Liberar memória intermediária

    // 3. Camada de Pooling
    Image* pool_output = poolForward(cnn->pool_layer, relu_output);
    freeImage(relu_output); // Liberar memória intermediária

    // 4. Achatar a saída do Pooling para a camada FC
    scalar_t* flattened_output = flattenImage(pool_output);
    freeImage(pool_output); // Liberar memória intermediária

    // 5. Camada Totalmente Conectada
    scalar_t* fc_output = fcForward(cnn->fc_layer, flattened_output);
    free(flattened_output); // Liberar memória intermediária

    // 6. Camada Softmax (para obter probabilidades de saída)
    scalar_t* softmax_output = softmaxForward(fc_output, cnn->fc_layer->output_size);
    free(fc_output); // Liberar memória intermediária

    return softmax_output; // Retorna as probabilidades finais
}