/* STM32 EcoCompress Firmware Template */
#include "main.h"
#include "ai_platform.h"
#include "network.h"
#include "network_data.h"

/* User includes */
#include <string.h>
#include <stdio.h>

/* AI model handler */
static ai_handle network = AI_HANDLE_NULL;
static ai_network_report network_info;

/* Input/Output buffers */
AI_ALIGNED(32) static ai_float in_data[AI_NETWORK_IN_1_SIZE];
AI_ALIGNED(32) static ai_float out_data[AI_NETWORK_OUT_1_SIZE];

/* Data buffer for windowing */
#define WINDOW_SIZE 128
#define NUM_FEATURES 13
static float sensor_buffer[WINDOW_SIZE][NUM_FEATURES];
static uint16_t buffer_index = 0;

/* Function prototypes */
void AI_Init(void);
void AI_Run(float* input_data, float* output_data);
void ProcessSensorData(float* new_sensor_reading);
void NormalizeData(float* data, uint16_t size);

/**
 * @brief Initialize AI model
 */
void AI_Init(void) {
    ai_error err;
    
    /* Create network instance */
    err = ai_network_create(&network, AI_NETWORK_DATA_CONFIG);
    if (err.type != AI_ERROR_NONE) {
        printf("AI Network creation failed\r\n");
        return;
    }
    
    /* Initialize network */
    if (!ai_network_init(network, &network_info)) {
        printf("AI Network initialization failed\r\n");
        return;
    }
    
    printf("AI Model initialized successfully\r\n");
    printf("Model info: %s\r\n", network_info.model_name);
}

/**
 * @brief Run AI inference
 */
void AI_Run(float* input_data, float* output_data) {
    ai_i32 batch;
    ai_buffer ai_input[AI_NETWORK_IN_NUM] = AI_NETWORK_IN;
    ai_buffer ai_output[AI_NETWORK_OUT_NUM] = AI_NETWORK_OUT;
    
    /* Set input buffer */
    ai_input[0].data = AI_HANDLE_PTR(input_data);
    ai_output[0].data = AI_HANDLE_PTR(output_data);
    
    /* Run inference */
    batch = ai_network_run(network, ai_input, ai_output);
    if (batch != 1) {
        printf("AI inference failed\r\n");
    }
}

/**
 * @brief Process incoming sensor data
 */
void ProcessSensorData(float* new_sensor_reading) {
    /* Add new reading to circular buffer */
    memcpy(sensor_buffer[buffer_index], new_sensor_reading, NUM_FEATURES * sizeof(float));
    buffer_index = (buffer_index + 1) % WINDOW_SIZE;
    
    /* Check if we have a full window */
    static uint16_t samples_collected = 0;
    samples_collected++;
    
    if (samples_collected >= WINDOW_SIZE) {
        /* Normalize the windowed data */
        NormalizeData((float*)sensor_buffer, WINDOW_SIZE * NUM_FEATURES);
        
        /* Run AI compression */
        AI_Run((float*)sensor_buffer, out_data);
        
        /* Process compressed output */
        printf("Compression complete\r\n");
        
        /* Reset for next window */
        samples_collected = WINDOW_SIZE / 2; /* 50% overlap */
    }
}

/**
 * @brief Normalize data (MinMax scaling)
 */
void NormalizeData(float* data, uint16_t size) {
    /* Simple normalization - you'll need to implement proper MinMax scaling */
    for (uint16_t i = 0; i < size; i++) {
        /* Placeholder normalization */
        data[i] = (data[i] - 0.0f) / 1.0f; /* Replace with actual min/max values */
    }
}

/**
 * @brief Main application entry point
 */
void EcoCompress_Main(void) {
    printf("EcoCompress STM32 Starting...\r\n");
    
    /* Initialize AI model */
    AI_Init();
    
    /* Simulation of sensor data input */
    float test_sensor_data[NUM_FEATURES] = {0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.9, 0.1, 0.7, 0.5, 25.0, 60.0, 15.0};
    
    /* Process sensor data */
    for (int i = 0; i < 200; i++) {
        /* Simulate varying sensor readings */
        for (int j = 0; j < NUM_FEATURES; j++) {
            test_sensor_data[j] += (float)(rand() % 100 - 50) / 1000.0f;
        }
        
        ProcessSensorData(test_sensor_data);
        HAL_Delay(10); /* 10ms delay between readings */
    }
}
