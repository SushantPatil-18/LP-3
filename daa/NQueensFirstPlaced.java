import java.util.*;

public class NQueensFirstPlaced {

    static int N;
    static int board[][];

    // Check if a queen can be placed at board[row][col]
    static boolean isSafe(int row, int col) {

        // Check this row on left side
        for (int i = 0; i < col; i++)
            if (board[row][i] == 1)
                return false;

        // Check upper diagonal
        for (int i = row, j = col; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 1)
                return false;

        // Check lower diagonal
        for (int i = row, j = col; j >= 0 && i < N; i++, j--)
            if (board[i][j] == 1)
                return false;

        return true;
    }

    // Backtracking function to place queens
    static boolean solveNQ(int col) {

        if (col == N)
            return true;

        // If column == first queen's column, skip it
        if (col == firstQueenCol) {
            return solveNQ(col + 1);
        }

        for (int i = 0; i < N; i++) {

            if (isSafe(i, col)) {
                board[i][col] = 1;

                if (solveNQ(col + 1))
                    return true;

                board[i][col] = 0; // BACKTRACK
            }
        }
        return false;
    }

    static int firstQueenRow, firstQueenCol;

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter value of N: ");
        N = sc.nextInt();

        board = new int[N][N];

        System.out.print("Enter row of first queen (0 to " + (N - 1) + "): ");
        firstQueenRow = sc.nextInt();

        System.out.print("Enter column of first queen (0 to " + (N - 1) + "): ");
        firstQueenCol = sc.nextInt();

        // Place the first queen
        board[firstQueenRow][firstQueenCol] = 1;

        if (solveNQ(0)) {
            System.out.println("\nFinal N-Queens Solution:");
            printBoard();
        } else {
            System.out.println("No solution exists for this configuration.");
        }
    }

    static void printBoard() {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                System.out.print(board[i][j] + " ");
            }
            System.out.println();
        }
    }
}
