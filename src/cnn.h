#ifndef CNN_H
#define CNN_H

// Definição de um tipo para float para facilitar futuras mudanças de precisão
typedef float scalar_t;

// Estrutura para representar uma imagem ou feature map
typedef struct {
    int depth;  // Número de canais (e.g., 1 para grayscale, 3 para RGB)
    int rows;   // Altura
    int cols;   // Largura
    scalar_t *data; // Dados da imagem/feature map (array unidimensional)
} Image;

// Funções auxiliares para manipulação de imagens/feature maps
Image* createImage(int depth, int rows, int cols);
void freeImage(Image *img);
scalar_t getPixel(Image *img, int d, int r, int c);
void setPixel(Image *img, int d, int r, int c, scalar_t value);

// Estrutura para a camada convolucional
typedef struct {
    int num_filters;    // Número de filtros
    int kernel_size;    // Tamanho do kernel (e.g., 3 para 3x3)
    int stride;         // Passo (stride)
    int padding;        // Preenchimento (padding)
    Image **filters;    // Array de kernels (cada kernel é uma Image)
    Image **biases;     // Array de vieses (cada bias é uma Image 1x1x1)
} ConvLayer;

// Estrutura para a camada de pooling
typedef struct {
    int pool_size;      // Tamanho da janela de pooling (e.g., 2 para 2x2)
    int stride;         // Passo (stride)
} PoolLayer;

// Funções para inicializar e liberar camadas
ConvLayer* createConvLayer(int num_filters, int kernel_size, int stride, int padding, int input_depth);
void freeConvLayer(ConvLayer *layer);
PoolLayer* createPoolLayer(int pool_size, int stride);
void freePoolLayer(PoolLayer *layer);

// Funções de forward pass para as camadas
Image* convForward(ConvLayer *layer, Image *input);
Image* reluForward(Image *input);
Image* poolForward(PoolLayer *layer, Image *input);

// Estrutura para a camada totalmente conectada
typedef struct {
    int input_size;     // Número de neurônios na entrada desta camada
    int output_size;    // Número de neurônios na saída desta camada
    scalar_t *weights;  // Pesos da camada (matriz input_size x output_size)
    scalar_t *biases;   // Vieses da camada (vetor output_size)
} FCLayer;

// Funções para inicializar e liberar a camada FC
FCLayer* createFCLayer(int input_size, int output_size);
void freeFCLayer(FCLayer *layer);

// Função de forward pass para a camada FC
scalar_t* fcForward(FCLayer *layer, scalar_t *input_vector);

// Função de ativação Softmax (para a camada de saída)
scalar_t* softmaxForward(scalar_t *input_vector, int size);

// Função para "achatar" uma Image em um vetor 1D
scalar_t* flattenImage(Image *img);

// Estrutura para a Rede Neural Convolucional completa
typedef struct {
    ConvLayer *conv_layer;
    PoolLayer *pool_layer;
    FCLayer *fc_layer;
    // Adicione mais camadas conforme sua arquitetura
} CNN;

// Funções para inicializar e liberar a CNN
CNN* createCNN(int input_depth, int input_rows, int input_cols, int num_classes);
void freeCNN(CNN* cnn);

// Função de forward pass para a CNN completa
scalar_t* cnnForward(CNN* cnn, Image* input_image);

// Função de perda
scalar_t crossEntropyLoss(scalar_t *predictions, int true_label_idx, int num_classes);

// Funções de backpropagation
scalar_t* softmaxBackward(scalar_t *predictions, int true_label_idx, int num_classes);
scalar_t* fcBackward(FCLayer *layer, scalar_t *d_output, scalar_t *input_vector, scalar_t *d_weights, scalar_t *d_biases);
Image* reluBackward(Image* input, Image* d_output);
Image* poolBackward(PoolLayer* layer, Image* input, Image* output, Image* d_output);
Image* convBackward(ConvLayer* layer, Image* input, Image* d_output, Image** d_filters, Image** d_biases);

#endif