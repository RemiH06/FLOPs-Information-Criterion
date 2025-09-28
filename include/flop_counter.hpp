#ifndef FLOP_COUNTER_HPP
#define FLOP_COUNTER_HPP

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>

// Estructura para almacenar información de operaciones
struct OperationInfo {
    std::string name;
    std::vector<int64_t> input_shapes;
    std::vector<int64_t> output_shapes;
    int64_t flops;
    double execution_time_ms;
    std::string library_source; // numpy, torch, tensorflow, etc.
};

// Clase principal del contador de FLOPs
class FLOPCounter {
private:
    std::atomic<int64_t> total_flops{0};
    std::vector<OperationInfo> operations;
    std::mutex operations_mutex;
    bool is_active{false};
    
    // Mapeo de funciones interceptadas
    std::unordered_map<std::string, std::function<int64_t(const std::vector<int64_t>&)>> flop_calculators;
    
    void initialize_calculators();
    
public:
    FLOPCounter();
    ~FLOPCounter();
    
    // Control del contador
    void start_counting();
    void stop_counting();
    void reset();
    
    // Obtener resultados
    int64_t get_total_flops() const;
    std::vector<OperationInfo> get_operations() const;
    void print_summary() const;
    
    // Función principal de intercepción
    void record_operation(
        const std::string& op_name,
        const std::vector<int64_t>& input_shapes,
        const std::vector<int64_t>& output_shapes,
        const std::string& library = "unknown",
        double execution_time = 0.0
    );
    
    // Calculadoras específicas de FLOPs
    int64_t calculate_matmul_flops(const std::vector<int64_t>& shapes) const;
    int64_t calculate_conv2d_flops(const std::vector<int64_t>& input_shape, 
                                   const std::vector<int64_t>& kernel_shape,
                                   const std::vector<int64_t>& output_shape) const;
    int64_t calculate_elementwise_flops(const std::vector<int64_t>& shapes) const;
    int64_t calculate_reduction_flops(const std::vector<int64_t>& input_shape,
                                     const std::vector<int64_t>& axes) const;
    
    // Singleton pattern
    static FLOPCounter& getInstance();
};

// Funciones C para la interfaz con Python
extern "C" {
    void* create_flop_counter();
    void destroy_flop_counter(void* counter);
    void start_counting(void* counter);
    void stop_counting(void* counter);
    void reset_counter(void* counter);
    int64_t get_total_flops(void* counter);
    void record_operation_c(void* counter, const char* op_name, 
                           int64_t* input_shapes, int input_dims,
                           int64_t* output_shapes, int output_dims,
                           const char* library, double execution_time);
    void print_summary(void* counter);
}

#endif // FLOP_COUNTER_HPP