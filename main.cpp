#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <iomanip>

using namespace std;

// ============================================================================
// HỆ THỐNG GIẢ LẬP XÉT NGHIỆM PCR (MOCK LAB ORACLE)
// ============================================================================
struct MockLab {
    vector<bool> true_status; // Trạng thái thực tế: true nếu dương tính, false nếu âm tính
    int test_count = 0;

    // Ngân sách sai số bị chặn (Cho bài toán 3.4.1)
    int fn_budget = 0; 
    int fp_budget = 0; 

    void reset(const vector<bool>& status) {
        true_status = status;
        test_count = 0;
        fn_budget = 0;
        fp_budget = 0;
    }

    // 1. Xét nghiệm lý tưởng (Không sai số)
    bool query_pool(const vector<int>& indices) {
        test_count++;
        for (int idx : indices) {
            if (true_status[idx]) return true; 
        }
        return false;
    }

    // 2. Xét nghiệm có sai số bị chặn (Bài toán 3.4.1)
    bool query_pool_noisy_bounded(const vector<int>& indices, double error_prob = 0.3) {
        test_count++;
        bool ideal_result = false;
        for (int idx : indices) {
            if (true_status[idx]) {
                ideal_result = true;
                break;
            }
        }

        static mt19937 rng(1337);
        uniform_real_distribution<double> dist(0.0, 1.0);

        if (ideal_result && fn_budget > 0 && dist(rng) < error_prob) {
            fn_budget--;
            return false; // Trả về Âm tính giả (FN)
        }
        if (!ideal_result && fp_budget > 0 && dist(rng) < error_prob) {
            fp_budget--;
            return true; // Trả về Dương tính giả (FP)
        }
        return ideal_result;
    }

    // 3. Xét nghiệm có sai số xác suất độc lập (Bài toán 3.4.2)
    bool query_pool_probabilistic(const vector<int>& indices, double alpha, double beta) {
        test_count++;
        bool ideal_result = false;
        for (int idx : indices) {
            if (true_status[idx]) {
                ideal_result = true;
                break;
            }
        }

        static mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist(0.0, 1.0);

        if (ideal_result) {
            if (dist(rng) < beta) return false; // Âm tính giả với xác suất beta
            return true;
        } else {
            if (dist(rng) < alpha) return true; // Dương tính giả với xác suất alpha
            return false;
        }
    }

    // 4. Xét nghiệm chịu ảnh hưởng bởi hiện tượng pha loãng mẫu (Bài toán 3.5)
    bool query_pool_dilution(const vector<int>& indices, double beta0, double theta, double k_slope, double V0) {
        test_count++;
        bool ideal_result = false;
        for (int idx : indices) {
            if (true_status[idx]) {
                ideal_result = true;
                break;
            }
        }
        if (!ideal_result) return false;

        // Tính toán xác suất âm tính giả beta(m) theo mô hình logit sinh học
        int m = indices.size();
        double exponent = -k_slope * ((V0 / m) - theta);
        double beta_m = beta0 + (1.0 - beta0) / (1.0 + exp(exponent));

        static mt19937 rng(random_device{}());
        uniform_real_distribution<double> dist(0.0, 1.0);

        if (dist(rng) < beta_m) return false; // Âm tính giả do nồng độ virus tụt dưới ngưỡng LoD
        return true;
    }
};

// ============================================================================
// TRÌNH SINH CA BỆNH THỬ NGHIỆM (TEST GENERATOR)
// ============================================================================
struct TestGenerator {
    static vector<bool> single_infected(int N, int& infected_idx) {
        vector<bool> status(N, false);
        mt19937 rng(random_device{}());
        uniform_int_distribution<int> dist(0, N - 1);
        infected_idx = dist(rng);
        status[infected_idx] = true;
        return status;
    }

    static vector<bool> k_infected(int N, int k) {
        vector<bool> status(N, false);
        mt19937 rng(random_device{}());
        vector<int> indices(N);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), rng);
        
        uniform_int_distribution<int> k_dist(1, k);
        int actual_k = k_dist(rng); 
        for (int i = 0; i < actual_k; ++i) {
            status[indices[i]] = true;
        }
        return status;
    }

    static vector<bool> consecutive_infected(int N, int k, int& start, int& length) {
        vector<bool> status(N, false);
        mt19937 rng(random_device{}());
        uniform_int_distribution<int> len_dist(1, k);
        length = len_dist(rng);
        uniform_int_distribution<int> start_dist(0, N - length);
        start = start_dist(rng);
        for (int i = start; i < start + length; ++i) {
            status[i] = true;
        }
        return status;
    }
};

// ============================================================================
// CHƯƠNG 3: TRIỂN KHAI CÁC THUẬT TOÁN TOÁN HỌC CỐT LÕI
// ============================================================================

// --- 3.1: Tìm kiếm nhị phân thích nghi (Trường hợp k = 1) ---
int adaptive_binary_search(int N, MockLab& lab) {
    int L = 0, R = N - 1;
    while (L < R) {
        int mid = L + (R - L) / 2;
        vector<int> pool;
        for (int i = L; i <= mid; ++i) pool.push_back(i);

        if (lab.query_pool(pool)) {
            R = mid; 
        } else {
            L = mid + 1; 
        }
    }
    return L;
}

// --- 3.2: Thuật toán Chia tách nhị phân tổng quát Hwang (Trường hợp k > 1) ---
int hwang_extract_one(const vector<int>& G, MockLab& lab, vector<int>& confirmed_negatives) {
    int L = 0, R = G.size() - 1;
    while (L < R) {
        int mid = L + (R - L) / 2;
        vector<int> pool(G.begin() + L, G.begin() + mid + 1);
        if (lab.query_pool(pool)) {
            // Những người từ mid+1 đến R tạm thời chưa xác định, nhưng phần ngoài pool này ở bước hiện tại là an toàn
            for (int i = mid + 1; i <= R; ++i) {
                confirmed_negatives.push_back(G[i]);
            }
            R = mid;
        } else {
            for (int i = L; i <= mid; ++i) {
                confirmed_negatives.push_back(G[i]);
            }
            L = mid + 1;
        }
    }
    return G[L];
}

vector<int> hwang_gbs(const vector<int>& U, int k, MockLab& lab) {
    if (U.empty() || k <= 0) return {};
    if (U.size() <= (size_t)k) {
        vector<int> infected;
        for (int person : U) {
            if (lab.query_pool({person})) {
                infected.push_back(person);
            }
        }
        return infected;
    }

    // Công thức tính m = 2^alpha tối ưu của Hwang
    double ratio = (double)(U.size() - k + 1) / k;
    int alpha = (ratio > 1.0) ? floor(log2(ratio)) : 0;
    size_t m = (size_t)1 << alpha;
    if (m > U.size()) m = U.size();

    vector<int> G(U.begin(), U.begin() + m);
    vector<int> rest(U.begin() + m, U.end());

    if (lab.query_pool(G)) {
        vector<int> confirmed_negatives;
        int infected_person = hwang_extract_one(G, lab, confirmed_negatives);
        vector<int> detected_infected = {infected_person};

        // Loại bỏ các phần tử đã biết chắc chắn âm tính hoặc dương tính
        vector<int> next_U;
        for (int person : U) {
            if (person == infected_person) continue;
            if (find(confirmed_negatives.begin(), confirmed_negatives.end(), person) != confirmed_negatives.end()) continue;
            next_U.push_back(person);
        }

        vector<int> sub_result = hwang_gbs(next_U, k - 1, lab);
        detected_infected.insert(detected_infected.end(), sub_result.begin(), sub_result.end());
        return detected_infected;
    } else {
        return hwang_gbs(rest, k, lab);
    }
}

// --- 3.3: Thuật toán quét cụm nhiễm liên tiếp (Cluster Testing) ---
vector<int> consecutive_cluster_testing(int N, int k, MockLab& lab) {
    // Đánh giá điều kiện ranh giới thích nghi quyết định chiến lược toàn cục
    int S1_bound = ceil((double)N / k) + 2 * ceil(log2(k));
    int S2_bound = 2 * ceil(log2(N));

    if (S1_bound <= S2_bound) {
        // Phương pháp 1: Quét khối cục bộ (Mật độ cao)
        int num_blocks = (N + k - 1) / k;
        vector<int> positive_blocks;

        for (int b = 0; b < num_blocks; ++b) {
            vector<int> pool;
            int start = b * k;
            int end = min(N, (b + 1) * k);
            for (int i = start; i < end; ++i) pool.push_back(i);

            if (lab.query_pool(pool)) {
                positive_blocks.push_back(b);
            }
        }

        if (positive_blocks.empty()) return {};

        int first_block = positive_blocks.front();
        int last_block = positive_blocks.back();

        // Nhị phân biên trái L
        int L_start = first_block * k;
        int L_end = min(N - 1, (first_block + 1) * k - 1);
        int L = L_start, R = L_end;
        while (L < R) {
            int mid = L + (R - L) / 2;
            vector<int> pool;
            for (int i = L; i <= mid; ++i) pool.push_back(i);
            if (lab.query_pool(pool)) R = mid;
            else L = mid + 1;
        }
        int left_boundary = L;

        // Nhị phân biên phải R
        int R_start = last_block * k;
        int R_end = min(N - 1, (last_block + 1) * k - 1);
        L = R_start, R = R_end;
        while (L < R) {
            int mid = L + (R - L) / 2;
            vector<int> pool;
            for (int i = mid + 1; i <= R; ++i) pool.push_back(i);
            if (lab.query_pool(pool)) L = mid + 1;
            else R = mid;
        }
        int right_boundary = L;

        vector<int> infected;
        for (int i = left_boundary; i <= right_boundary; ++i) infected.push_back(i);
        return infected;
    } else {
        // Phương pháp 2: Dò biên nhị phân toàn cục (Mật độ thấp)
        int L = 0, R = N - 1;
        while (L < R) {
            int mid = L + (R - L) / 2;
            vector<int> pool;
            for (int i = 0; i <= mid; ++i) pool.push_back(i);
            if (lab.query_pool(pool)) R = mid;
            else L = mid + 1;
        }
        int left_boundary = L;

        L = left_boundary, R = N - 1;
        while (L < R) {
            int mid = L + (R - L) / 2;
            vector<int> pool;
            for (int i = mid + 1; i <= N - 1; ++i) pool.push_back(i);
            if (lab.query_pool(pool)) L = mid + 1;
            else R = mid;
        }
        int right_boundary = L;

        // Kiểm chứng tính chính xác ranh giới
        if (left_boundary == N - 1 && !lab.query_pool({N - 1})) return {};

        vector<int> infected;
        for (int i = left_boundary; i <= right_boundary; ++i) infected.push_back(i);
        return infected;
    }
}

// --- 3.4.1: Tìm kiếm thích nghi kháng nhiễu giới hạn (Bounded Noise) ---
int robust_binary_search_bounded(int N, int a, int b, MockLab& lab) {
    int L = 0, R = N - 1;
    lab.fn_budget = a;
    lab.fp_budget = b;

    while (L < R) {
        int mid = L + (R - L) / 2;
        vector<int> pool;
        for (int i = L; i <= mid; ++i) pool.push_back(i);

        int pos_votes = 0;
        int neg_votes = 0;
        
        // Cơ chế bỏ phiếu thích nghi cải tiến
        while (pos_votes < b + 1 && neg_votes < a + 1) {
            bool res = lab.query_pool_noisy_bounded(pool);
            if (res) pos_votes++;
            else neg_votes++;
        }

        if (pos_votes >= b + 1) R = mid; 
        else L = mid + 1; 
    }
    return L;
}

// --- 3.4.2: Thuật toán kiểm định cập nhật xác suất Bayes (Probabilistic Noise) ---
int bayesian_group_testing(int N, double alpha, double beta, MockLab& lab, double p0 = 0.02) {
    vector<double> p(N, p0);
    int iterations = 0;
    const double epsilon_negative = 1e-4; // Ngưỡng an toàn âm tính
    const double epsilon_positive = 0.95; // Ngưỡng tin cậy dương tính

    mt19937 rng(105);

    while (iterations < 150) {
        // Thiết kế nhóm gộp thông minh dựa trên entropy: Chọn nhóm có xác suất chứa F0 gần 50% nhất
        vector<int> pool;
        double current_pool_neg_prob = 1.0;
        vector<int> candidates(N);
        iota(candidates.begin(), candidates.end(), 0);
        shuffle(candidates.begin(), candidates.end(), rng);

        for (int idx : candidates) {
            if (p[idx] > epsilon_negative && p[idx] < epsilon_positive) {
                if (current_pool_neg_prob * (1.0 - p[idx]) >= 0.5) {
                    pool.push_back(idx);
                    current_pool_neg_prob *= (1.0 - p[idx]);
                }
            }
        }

        if (pool.empty()) break;

        // Xét nghiệm và cập nhật Bayes
        bool Y = lab.query_pool_probabilistic(pool, alpha, beta);
        double P_G_positive = 1.0 - current_pool_neg_prob;
        double P_Y1 = (1.0 - beta) * P_G_positive + alpha * (1.0 - P_G_positive);
        double P_Y0 = beta * P_G_positive + (1.0 - alpha) * (1.0 - P_G_positive);

        for (int idx : pool) {
            double P_G_minus_i_pos = 1.0 - (current_pool_neg_prob / (1.0 - p[idx]));
            if (Y) {
                double P_Y1_given_Xi1 = 1.0 - beta;
                p[idx] = (P_Y1_given_Xi1 * p[idx]) / P_Y1;
            } else {
                double P_Y0_given_Xi1 = beta;
                p[idx] = (P_Y0_given_Xi1 * p[idx]) / P_Y0;
            }
            // Giới hạn chống tràn số học
            p[idx] = max(1e-6, min(0.9999, p[idx]));
        }
        iterations++;
    }

    // Trả về chỉ số có xác suất dương tính cao nhất
    return distance(p.begin(), max_element(p.begin(), p.end()));
}

// --- 3.5: Thuật toán tối ưu hóa pha loãng mẫu (Dorfman 2-Stage + Dilution) ---
int optimize_pool_size_dilution(double beta0, double theta, double k_slope, double V0, double alpha, double p) {
    int best_m = 1;
    double min_cost = 99999.0;

    // Quét số học (Grid Search) tìm m tối ưu
    for (int m = 1; m <= 32; ++m) {
        double exponent = -k_slope * ((V0 / m) - theta);
        double beta_m = beta0 + (1.0 - beta0) / (1.0 + exp(exponent));
        double P_positive = (1.0 - pow(1.0 - p, m)) * (1.0 - beta_m) + pow(1.0 - p, m) * alpha;
        double cost = (1.0 / m) + P_positive;

        if (cost < min_cost && m <= 10) { // Chặn lâm sàng m <= 10 để bảo toàn độ nhạy lâm sàng
            min_cost = cost;
            best_m = m;
        }
    }
    return best_m;
}

// ============================================================================
// CHƯƠNG TRÌNH KIỂM CHỨNG TOÀN DIỆN (VALIDATION SUITE)
// ============================================================================
void run_validation_suite() {
    MockLab lab;
    cout << "==========================================================================\n";
    cout << "             BAT DAU CHUONG TRINH KIEM CHUNG TOAN MO HINH HOA              \n";
    cout << "==========================================================================\n\n";

    // --- KIỂM CHỨNG 3.1 ---
    cout << "--- KIEM CHUNG TRUONG HOP 1 (k = 1) ---\n";
    vector<int> test_N = {1000, 10000, 100000};
    for (int N : test_N) {
        int true_idx = -1;
        auto population = TestGenerator::single_infected(N, true_idx);
        lab.reset(population);
        int detected_idx = adaptive_binary_search(N, lab);
        cout << "N = " << setw(6) << N << " | Thuc: " << setw(5) << true_idx 
             << " | Doan: " << setw(5) << detected_idx << " | So test: " << setw(2) << lab.test_count;
        cout << (true_idx == detected_idx ? " [CHINH XAC]" : " [THAT BAI]") << endl;
    }
    cout << "\n";

    // --- KIỂM CHỨNG 3.2 ---
    cout << "--- KIEM CHUNG TRUONG HOP 2 (Hwang GBS cho k > 1) ---\n";
    vector<pair<int, int>> test_cases_32 = {{1000, 10}, {10000, 50}, {100000, 200}};
    for (auto tc : test_cases_32) {
        int N = tc.first;
        int k = tc.second;
        auto population = TestGenerator::k_infected(N, k);
        lab.reset(population);

        vector<int> U(N);
        iota(U.begin(), U.end(), 0);
        vector<int> detected = hwang_gbs(U, k, lab);
        sort(detected.begin(), detected.end());

        vector<int> actual;
        for (int i = 0; i < N; ++i) {
            if (population[i]) actual.push_back(i);
        }
        cout << "N = " << setw(6) << N << ", k = " << setw(3) << k 
             << " | Thuc co: " << setw(2) << actual.size() << " ca"
             << " | Doan: " << setw(2) << detected.size() << " ca"
             << " | So test: " << setw(4) << lab.test_count;
        cout << (detected == actual ? " [CHINH XAC]" : " [THAT BAI]") << endl;
    }
    cout << "\n";

    // --- KIỂM CHỨNG 3.3 ---
    cout << "--- KIEM CHUNG TRUONG HOP 3 (Cum lien tiep - Cluster Testing) ---\n";
    vector<pair<int, int>> test_cases_33 = {{1000, 10}, {10000, 50}, {100000, 200}};
    for (auto tc : test_cases_33) {
        int N = tc.first;
        int k = tc.second;
        int true_start = -1, true_len = -1;
        auto population = TestGenerator::consecutive_infected(N, k, true_start, true_len);
        lab.reset(population);

        vector<int> detected = consecutive_cluster_testing(N, k, lab);
        vector<int> actual;
        for (int i = 0; i < N; ++i) {
            if (population[i]) actual.push_back(i);
        }
        cout << "N = " << setw(6) << N << ", k = " << setw(3) << k 
             << " | Cum thuc: [" << setw(5) << true_start << " -> " << setw(5) << true_start + true_len - 1 << "]"
             << " | So test tieu thu: " << setw(3) << lab.test_count;
        cout << (detected == actual ? " [CHINH XAC]" : " [THAT BAI]") << endl;
    }
    cout << "\n";

    // --- KIỂM CHỨNG 3.4.1 ---
    cout << "--- KIEM CHUNG BAI TOAN 3.4.1 (Co nhieu gioi han) ---\n";
    int N_noisy = 1000;
    vector<pair<int, int>> noise_bounds = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 3}};
    for (auto bound : noise_bounds) {
        int a = bound.first;
        int b = bound.second;
        int true_idx = -1;
        auto population = TestGenerator::single_infected(N_noisy, true_idx);
        lab.reset(population);

        int detected_idx = robust_binary_search_bounded(N_noisy, a, b, lab);
        cout << "N = " << N_noisy << " | Loi (FN, FP) <= (" << a << ", " << b << ")"
             << " | Thuc: " << setw(4) << true_idx << " | Doan: " << setw(4) << detected_idx
             << " | So test: " << setw(3) << lab.test_count;
        cout << (true_idx == detected_idx ? " [KHANG NHIEU THANH CONG]" : " [THAT BAI]") << endl;
    }
    cout << "\n";

    // --- KIỂM CHỨNG 3.4.2 ---
    cout << "--- KIEM CHUNG BAI TOAN 3.4.2 (Nhieu xac suat Bayes - 9 truong hop) ---\n";
    int N_prob = 1000;
    vector<pair<double, double>> prob_cases = {
        {0.001, 0.001}, {0.001, 0.01}, {0.001, 0.2},
        {0.01, 0.001},  {0.01, 0.01},  {0.01, 0.2},
        {0.2, 0.001},   {0.2, 0.01},   {0.2, 0.2}
    };
    for (int i = 0; i < 9; ++i) {
        double alpha = prob_cases[i].first;
        double beta = prob_cases[i].second;
        int true_idx = -1;
        auto population = TestGenerator::single_infected(N_prob, true_idx);
        lab.reset(population);

        int detected_idx = bayesian_group_testing(N_prob, alpha, beta, lab);
        cout << "TH " << i + 1 << " | (alpha, beta) = (" << setw(5) << alpha << "; " << setw(5) << beta << ")"
             << " | Thuc: " << setw(3) << true_idx << " | Doan: " << setw(3) << detected_idx
             << " | So test: " << setw(3) << lab.test_count;
        cout << (true_idx == detected_idx ? " [CHINH XAC]" : " [THAT BAI]") << endl;
    }
    cout << "\n";

    // --- KIỂM CHỨNG 3.5 ---
    cout << "--- KIEM CHUNG TRUONG HOP 5 (Pha loang mau sinh hoc RT-PCR) ---\n";
    int N_dil = 10000;
    double p = 0.01; 
    double beta0 = 0.01;  
    double theta = 15.0; 
    double k_slope = 0.5; 
    double V0 = 100.0;   
    double alpha = 0.005;

    int m_star = optimize_pool_size_dilution(beta0, theta, k_slope, V0, alpha, p);
    cout << "Voi p = " << p * 100 << "%, beta0 = " << beta0 << ", theta = " << theta << ", LoD = " << V0 << "\n";
    cout << "-> Kich thuoc nhom gop toi uu hoa tu dong m* = " << m_star << " người/nhóm." << endl;
    cout << "==========================================================================\n";
}

int main() {
    run_validation_suite();
    return 0;
}
