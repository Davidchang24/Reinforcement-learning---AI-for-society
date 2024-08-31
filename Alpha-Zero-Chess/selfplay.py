import argparse
import chess
import chess.pgn
import MCTS
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import AlphaZeroNetwork
import time
import os
import pickle
import encoder
import matplotlib.pyplot as plt

cuda = True if torch.cuda.is_available() else False

class SelfPlayDataset(Dataset):
    def __init__(self, game_data):
        self.game_data = game_data

    def __len__(self):
        return len(self.game_data)

    def __getitem__(self, idx):
        fen, move = self.game_data[idx]
        board = chess.Board(fen)
        winner = 1 if board.turn == chess.WHITE else -1  # Simplified winner encoding
        position, policy, value, mask = encoder.encodeTrainingPoint(board, move, winner)
        return {
            'position': torch.from_numpy(position),
            'policy': torch.Tensor([policy]).long(),
            'value': torch.Tensor([value]),
            'mask': torch.from_numpy(mask)
        }

def tolist(move_generator):
    moves = []
    for move in move_generator:
        moves.append(move)
    return moves

def play_game(model, num_rollouts, num_threads, verbose):
    model.eval()
    board = chess.Board()
    game_data = []
    moves = []

    while not board.is_game_over():
        starttime = time.perf_counter()

        with torch.no_grad():
            root = MCTS.Root(board, model)
            for i in range(num_rollouts):
                root.parallelRollouts(board.copy(), model, num_threads)

        endtime = time.perf_counter()
        elapsed = endtime - starttime

        Q = root.getQ()
        N = root.getN()
        nps = N / elapsed
        same_paths = root.same_paths

        # Print MCTS statistics after rollouts
        if verbose:
            print(f"\nMCTS Statistics after {num_rollouts} rollouts:")
            print(root.getStatisticsString())
            print(f'Total rollouts: {int(N)} | Q-value: {Q:.3f} | Duplicate paths: {same_paths} | '
                  f'Elapsed time: {elapsed:.2f} s | Nodes per second (nps): {nps:.2f}')

        # Select the move with the highest visit count
        edge = root.maxNSelect()
        bestmove = edge.getMove()

        # Print the selected move
        if verbose:
            print(f"Selected move: {bestmove} | FEN: {board.fen()}")

        # Store game data and make the move
        game_data.append((board.fen(), bestmove))
        moves.append(bestmove)
        board.push(bestmove)

    result = board.result()

    # Convert moves to PGN format
    game = chess.pgn.Game()
    node = game.add_variation(moves[0])
    for move in moves[1:]:
        node = node.add_variation(move)

    game.headers["Result"] = result
    pgn = str(game)

    print(f"\nFinal Game PGN:\n{pgn}")

    return game_data, result

def train_model(alphaZeroNet, game_data, num_epochs):
    train_ds = SelfPlayDataset(game_data)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)

    optimizer = optim.Adam(alphaZeroNet.parameters())
    mseLoss = nn.MSELoss()

    print('Starting training')

    losses = {'value': [], 'policy': [], 'total': []}

    for epoch in range(num_epochs):
        alphaZeroNet.train()
        for iter_num, data in enumerate(train_loader):
            optimizer.zero_grad()

            if cuda:
                position = data['position'].cuda()
                valueTarget = data['value'].cuda()
                policyTarget = data['policy'].cuda()
            else:
                position = data['position']
                valueTarget = data['value']
                policyTarget = data['policy']

            valueLoss, policyLoss = alphaZeroNet(position, valueTarget=valueTarget, policyTarget=policyTarget)
            loss = valueLoss + policyLoss

            loss.backward()
            optimizer.step()

            losses['value'].append(float(valueLoss))
            losses['policy'].append(float(policyLoss))
            losses['total'].append(float(loss))

            message = 'Epoch {:03} | Step {:05} / {:05} | Value loss {:0.5f} | Policy loss {:0.5f}'.format(
                epoch, iter_num, len(train_loader), float(valueLoss), float(policyLoss))
            print(message)

    return losses

def save_game_data(game_data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(game_data, f)

def load_game_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def evaluate_models(model1, model2, num_games, num_rollouts, num_threads, verbose):
    model1.eval()
    model2.eval()
    model1_wins, model2_wins, draws = 0, 0, 0

    for game_idx in range(num_games):
        _, result = play_game(model1, num_rollouts, num_threads, verbose)
        if result == '1-0':
            model1_wins += 1
        elif result == '0-1':
            model2_wins += 1
        else:
            draws += 1

    return model1_wins, model2_wins, draws

def plot_losses(losses, save_dir):
    plt.figure()
    plt.plot(losses['value'], label='Value Loss')
    plt.plot(losses['policy'], label='Policy Loss')
    plt.plot(losses['total'], label='Total Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses Over Steps')
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.show()

def main(modelFile, num_rollouts, num_threads, num_epochs, num_games, num_cycles, save_dir, verbose):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    alphaZeroNet = AlphaZeroNetwork.AlphaZeroNet(20, 256)
    if cuda:
        alphaZeroNet.cuda()

    # If modelFile is provided, load it; otherwise, start with a fresh model
    if modelFile:
        if cuda:
            weights = torch.load(modelFile)
        else:
            weights = torch.load(modelFile, map_location=torch.device('cpu'))
        alphaZeroNet.load_state_dict(weights)
        print("Loaded model from file:", modelFile)
    else:
        print("No model file provided; starting from scratch.")

    alphaZeroNet.eval()

    all_losses = {'value': [], 'policy': [], 'total': []}

    for cycle in range(num_cycles):
        print(f'Starting cycle {cycle + 1}/{num_cycles}')
        for game_idx in range(num_games):
            game_data, result = play_game(alphaZeroNet, num_rollouts, num_threads, verbose)
            game_file = os.path.join(save_dir, f'cycle_{cycle}_game_{game_idx}.pkl')
            save_game_data(game_data, game_file)

            print(f'Cycle {cycle}, Game {game_idx} finished with result: {result}')

            losses = train_model(alphaZeroNet, game_data, num_epochs)

            all_losses['value'].extend(losses['value'])
            all_losses['policy'].extend(losses['policy'])
            all_losses['total'].extend(losses['total'])

            model_save_path = os.path.join(save_dir, f'cycle_{cycle}_model_{game_idx}.pt')
            torch.save(alphaZeroNet.state_dict(), model_save_path)

            if game_idx > 0:
                previous_model = AlphaZeroNetwork.AlphaZeroNet(20, 256)  # Match the architecture
                previous_model.load_state_dict(torch.load(os.path.join(save_dir, f'cycle_{cycle}_model_{game_idx - 1}.pt')))
                if cuda:
                    previous_model.cuda()

                model1_wins, model2_wins, draws = evaluate_models(previous_model, alphaZeroNet, 10, num_rollouts, num_threads, verbose)

                print(f'Previous model wins: {model1_wins}, Current model wins: {model2_wins}, Draws: {draws}')

                if model2_wins > model1_wins:
                    print('Current model is better. Keeping current model.')
                    torch.save(alphaZeroNet.state_dict(), model_save_path)
                else:
                    print('Previous model is better. Reverting to previous model.')
                    alphaZeroNet.load_state_dict(previous_model.state_dict())
                    model_save_path = os.path.join(save_dir, f'cycle_{cycle}_model_{game_idx}.pt')
                    torch.save(alphaZeroNet.state_dict(), model_save_path)

    plot_losses(all_losses, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage='Self-play and train the model.')
    parser.add_argument('--model', help='Path to model (.pt) file.', default=None)
    parser.add_argument('--rollouts', type=int, help='The number of rollouts on computers turn', default=20)
    parser.add_argument('--threads', type=int, help='Number of threads used per rollout', default=10)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train on each game', default=10)
    parser.add_argument('--games', type=int, help='Number of self-play games to generate', default=10)
    parser.add_argument('--cycles', type=int, help='Number of self-play and training cycles', default=3)
    parser.add_argument('--save_dir', help='Directory to save games and models', default='selfplay_data')
    parser.add_argument('--verbose', help='Print search statistics', action='store_true', default=True)  # Set default to True
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args.model, args.rollouts, args.threads, args.epochs, args.games, args.cycles, args.save_dir, args.verbose)

