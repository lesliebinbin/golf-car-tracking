#include <iostream>
#include <torch/torch.h>

struct Net : torch::nn::Module {
  Net() {
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = x.reshape({-1, 784});
    x = torch::relu(fc1->forward(x));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = fc3->forward(x);
    return torch::log_softmax(x, 1);
  }

  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

template <typename DataLoader, typename Device>
void test(std::shared_ptr<Net> net, DataLoader &test_loader, Device device) {
  net->eval();
  torch::NoGradGuard no_grad;

  double test_loss = 0.0;
  std::int64_t correct = 0;
  std::int64_t total = 0;

  for (const auto &batch : test_loader) {
    auto data = batch.data.to(device);
    auto targets = batch.target.to(device);

    auto output = net->forward(data);
    auto loss = torch::nll_loss(output, targets);

    test_loss += loss.template item<double>() * data.size(0);

    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
    total += data.size(0);
  }

  test_loss /= total;

  std::cout << "Test set: Average loss: " << test_loss
            << " | Accuracy: " << correct << "/" << total << " ("
            << (100.0 * correct / total) << "%)\n";
}

void train() {
  auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU")
            << std::endl;
  auto net = std::make_shared<Net>();
  net->to(device);

  auto train_data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./train/MNIST/raw",
                                   torch::data::datasets::MNIST::Mode::kTrain)
          .map(torch::data::transforms::Stack()),
      64);

  auto test_data_loader = torch::data::make_data_loader(
      torch::data::datasets::MNIST("./test/MNIST/raw",
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Stack()),
      64);

  auto optimizer =
      torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(1e-3));

  for (size_t epoch = 0; epoch < 100; epoch++) {
    net->train();
    size_t batch_index = 0;

    for (auto &batch : *train_data_loader) {
      auto data = batch.data.to(device);
      auto targets = batch.target.to(device);

      optimizer.zero_grad();
      auto output = net->forward(data);
      auto loss = torch::nll_loss(output, targets);
      loss.backward();
      optimizer.step();

      if (++batch_index % 100 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        torch::save(net, "net.pt");
      }
    }

    test(net, *test_data_loader, device);
  }
}

int main() {
  train();
  return 0;
}
