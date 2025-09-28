#include "../include/flop_counter.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

FLOPCounter::FLOPCounter() {
    initialize_calculators();
}

FLOPCounter::~FLOPCounter() {
    // Limpieza automática
}

FLOPCounter& FLOPCounter::getInstance() {
    static FLOPCounter instance;
    return instance;
}

void FLOPCounter::initialize_calculators() {
    // Matmul/Dot product
    flop_calculators["matmul"] = [this](const std::vector<int64_t>& shapes) {
        return calculate_matmul_flops(shapes);
    };
    
    flop_calculators["dot"] = flop_calculators["matmul"];
    flop_calculators["mm"] = flop_calculators["matmul"];
    flop_calculators["bmm"] = flop_calculators["matmul"];
    
    // Operaciones elementwise
    flop_calculators["add"] = [this](const std::vector<int64_t>& shapes) {
        return calculate_elementwise_flops(shapes);
    };
    flop_calculators["sub"] = flop_calculators["add"];
    flop_calculators["subtract"] = flop_calculators["add"];
    flop_calculators["mul"] = flop_calculators["add"];
    flop_calculators["multiply"] = flop_calculators["add"];
    flop_calculators["div"] = flop_calculators["add"];
    flop_calculators["divide"] = flop_calculators["add"];
    flop_calculators["pow"] = flop_calculators["add"];
    flop_calculators["power"] = flop_calculators["add"];
    flop_calculators["sqrt"] = flop_calculators["add"];
    flop_calculators["exp"] = flop_calculators["add"];
    flop_calculators["log"] = flop_calculators["add"];
    flop_calculators["sin"] = flop_calculators["add"];
    flop_calculators["cos"] = flop_calculators["add"];
    flop_calculators["tanh"] = flop_calculators["add"];
    flop_calculators["relu"] = flop_calculators["add"];
    flop_calculators["sigmoid"] = flop_calculators["add"];
    flop_calculators["softmax"] = flop_calculators["add"];
    
    // Operaciones de reducción
    flop_calculators["sum"] = [this](const std::vector<int64_t>& shapes) {
        return calculate_elementwise_flops(shapes);
    };
    flop_calculators["mean"] = flop_calculators["sum"];
    flop_calculators["max"] = flop_calculators["sum"];
    flop_calculators["min"] = flop_calculators["sum"];
    flop_calculators["reduce_sum"] = flop_calculators["sum"];
    flop_calculators["reduce_mean"] = flop_calculators["sum"];
    flop_calculators["reduce_max"] = flop_calculators["sum"];
    flop_calculators["reduce_min"] = flop_calculators["sum"];
    
    // Convoluciones (se manejan por separado)
    flop_calculators["conv2d"] = [](const std::vector<int64_t>&) { return 0; }; // Manejado especialmente
    flop_calculators["conv1d"] = [](const std::vector<int64_t>&) { return 0; };
    flop_calculators["conv3d"] = [](const std::vector<int64_t>&) { return 0; };
    
    // Operaciones de PyTorch functional
    flop_calculators["F.linear"] = flop_calculators["matmul"];
    flop_calculators["F.relu"] = flop_calculators["add"];
    flop_calculators["F.softmax"] = flop_calculators["add"];
    flop_calculators["F.dropout"] = flop_calculators["add"];
    flop_calculators["linear"] = flop_calculators["matmul"];
}

void FLOPCounter::start_counting() {
    std::lock_guard<std::mutex> lock(operations_mutex);
    is_active = true;
}

void FLOPCounter::stop_counting() {
    std::lock_guard<std::mutex> lock(operations_mutex);
    is_active = false;
}

void FLOPCounter::reset() {
    std::lock_guard<std::mutex> lock(operations_mutex);
    total_flops.store(0);
    operations.clear();
}

int64_t FLOPCounter::get_total_flops() const {
    return total_flops.load();
}

std::vector<OperationInfo> FLOPCounter::get_operations() const {
    std::lock_guard<std::mutex> lock(operations_mutex);
    return operations;
}

void FLOPCounter::record_operation(
    const std::string& op_name,
    const std::vector<int64_t>& input_shapes,
    const std::vector<int64_t>& output_shapes,
    const std::string& library,
    double execution_time) {
    
    if (!is_active) return;
    
    int64_t op_flops = 0;
    
    // Buscar calculadora específica
    auto calc_it = flop_calculators.find(op_name);
    if (calc_it != flop_calculators.end()) {
        op_flops = calc_it->second(input_shapes);
    } else {
        // Estimación genérica basada en el tamaño de salida
        int64_t output_elements = 1;
        for (auto dim : output_shapes) {
            output_elements *= dim;
        }
        op_flops = output_elements; // Al menos 1 FLOP por elemento de salida
    }
    
    // Registrar operación
    OperationInfo op_info;
    op_info.name = op_name;
    op_info.input_shapes = input_shapes;
    op_info.output_shapes = output_shapes;
    op_info.flops = op_flops;
    op_info.execution_time_ms = execution_time;
    op_info.library_source = library;
    
    {
        std::lock_guard<std::mutex> lock(operations_mutex);
        operations.push_back(op_info);
    }
    
    total_flops.fetch_add(op_flops);
}

int64_t FLOPCounter::calculate_matmul_flops(const std::vector<int64_t>& shapes) const {
    if (shapes.size() < 2) return 0;
    
    // Para matmul, necesitamos interpretar las formas de entrada
    // Asumimos que tenemos formas concatenadas: [shape1_dims..., shape2_dims...]
    // Dividimos por la mitad para obtener las dos matrices
    size_t mid = shapes.size() / 2;
    
    std::vector<int64_t> shape1(shapes.begin(), shapes.begin() + mid);
    std::vector<int64_t> shape2(shapes.begin() + mid, shapes.end());
    
    if (shape1.empty() || shape2.empty()) return 0;
    
    // Obtener dimensiones para multiplicación de matrices
    // Para A (m x k) * B (k x n) = C (m x n)
    int64_t m = shape1.size() >= 2 ? shape1[shape1.size()-2] : 1;
    int64_t k = shape1.size() >= 1 ? shape1[shape1.size()-1] : 1;
    int64_t n = shape2.size() >= 1 ? shape2[shape2.size()-1] : 1;
    
    // Calcular batch size
    int64_t batch_size = 1;
    for (size_t i = 0; i < shape1.size() - 2; ++i) {
        batch_size *= shape1[i];
    }
    
    return 2 * batch_size * m * k * n; // 2 FLOPs por multiplicación-suma
}

int64_t FLOPCounter::calculate_conv2d_flops(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& kernel_shape,
    const std::vector<int64_t>& output_shape) const {
    
    if (input_shape.size() < 4 || kernel_shape.size() < 4 || output_shape.size() < 4) {
        return 0;
    }
    
    // input: [batch, channels_in, h_in, w_in]
    // kernel: [channels_out, channels_in, k_h, k_w]
    // output: [batch, channels_out, h_out, w_out]
    
    int64_t batch_size = output_shape[0];
    int64_t channels_out = output_shape[1];
    int64_t h_out = output_shape[2];
    int64_t w_out = output_shape[3];
    
    int64_t channels_in = kernel_shape[1];
    int64_t k_h = kernel_shape[2];
    int64_t k_w = kernel_shape[3];
    
    return batch_size * channels_out * h_out * w_out * channels_in * k_h * k_w * 2;
}

int64_t FLOPCounter::calculate_elementwise_flops(const std::vector<int64_t>& shapes) const {
    if (shapes.empty()) return 0;
    
    // Para operaciones elementwise, contamos todos los elementos
    int64_t total_elements = 1;
    for (auto dim : shapes) {
        total_elements *= dim;
    }
    return total_elements;
}

int64_t FLOPCounter::calculate_reduction_flops(
    const std::vector<int64_t>& input_shape,
    const std::vector<int64_t>& axes) const {
    
    int64_t total_elements = 1;
    for (auto dim : input_shape) {
        total_elements *= dim;
    }
    return total_elements;
}

void FLOPCounter::print_summary() const {
    std::lock_guard<std::mutex> lock(operations_mutex);
    
    std::cout << "\n=== FLOP Counter Summary ===" << std::endl;
    std::cout << "Total FLOPs: " << std::setprecision(3) << std::scientific 
              << static_cast<double>(total_flops.load()) << std::endl;
    std::cout << "Total Operations: " << operations.size() << std::endl;
    
    // Agrupar por tipo de operación
    std::unordered_map<std::string, int64_t> flops_by_op;
    std::unordered_map<std::string, int> count_by_op;
    std::unordered_map<std::string, double> time_by_op;
    
    for (const auto& op : operations) {
        flops_by_op[op.name] += op.flops;
        count_by_op[op.name]++;
        time_by_op[op.name] += op.execution_time_ms;
    }
    
    std::cout << "\nBreakdown by operation:" << std::endl;
    std::cout << std::setw(15) << "Operation" << std::setw(8) << "Count" 
              << std::setw(15) << "Total FLOPs" << std::setw(12) << "Avg FLOPs"
              << std::setw(12) << "Total Time" << std::endl;
    std::cout << std::string(72, '-') << std::endl;
    
    for (const auto& pair : flops_by_op) {
        const std::string& op_name = pair.first;
        int64_t total_op_flops = pair.second;
        int op_count = count_by_op[op_name];
        double total_time = time_by_op[op_name];
        
        std::cout << std::setw(15) << op_name 
                  << std::setw(8) << op_count
                  << std::setw(15) << std::scientific << static_cast<double>(total_op_flops)
                  << std::setw(12) << std::scientific 
                  << static_cast<double>(total_op_flops) / op_count
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << total_time << "ms" << std::endl;
    }
    std::cout << std::endl;
}

// Implementación de funciones C para interfaz con Python
extern "C" {
    void* create_flop_counter() {
        return new FLOPCounter();
    }
    
    void destroy_flop_counter(void* counter) {
        delete static_cast<FLOPCounter*>(counter);
    }
    
    void start_counting(void* counter) {
        static_cast<FLOPCounter*>(counter)->start_counting();
    }
    
    void stop_counting(void* counter) {
        static_cast<FLOPCounter*>(counter)->stop_counting();
    }
    
    void reset_counter(void* counter) {
        static_cast<FLOPCounter*>(counter)->reset();
    }
    
    int64_t get_total_flops(void* counter) {
        return static_cast<FLOPCounter*>(counter)->get_total_flops();
    }
    
    void record_operation_c(void* counter, const char* op_name,
                           int64_t* input_shapes, int input_dims,
                           int64_t* output_shapes, int output_dims,
                           const char* library, double execution_time) {
        
        std::vector<int64_t> input_vec(input_shapes, input_shapes + input_dims);
        std::vector<int64_t> output_vec(output_shapes, output_shapes + output_dims);
        
        static_cast<FLOPCounter*>(counter)->record_operation(
            std::string(op_name), input_vec, output_vec, 
            std::string(library), execution_time);
    }
    
    void print_summary(void* counter) {
        static_cast<FLOPCounter*>(counter)->print_summary();
    }
}