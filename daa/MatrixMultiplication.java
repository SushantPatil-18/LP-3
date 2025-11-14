import java.util.*;

public class MatrixMultiplication {

    // -------------------- NORMAL MATRIX MULTIPLICATION --------------------
    public static int[][] normalMultiply(int[][] A, int[][] B) {
        int r1 = A.length;
        int c1 = A[0].length;
        int c2 = B[0].length;

        int[][] result = new int[r1][c2];

        for (int i = 0; i < r1; i++) {
            for (int j = 0; j < c2; j++) {
                for (int k = 0; k < c1; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    // -------------------- MULTITHREADED (THREAD PER ROW) --------------------
    static class RowMultiplier extends Thread {
        int row;
        int[][] A, B, result;

        RowMultiplier(int row, int[][] A, int[][] B, int[][] result) {
            this.row = row;
            this.A = A;
            this.B = B;
            this.result = result;
        }

        @Override
        public void run() {
            int colsB = B[0].length;
            int colsA = A[0].length;

            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[row][j] += A[row][k] * B[k][j];
                }
            }
        }
    }

    public static int[][] threadedMultiply(int[][] A, int[][] B) {
        int r1 = A.length;
        int c2 = B[0].length;

        int[][] result = new int[r1][c2];
        Thread[] threads = new Thread[r1];

        for (int i = 0; i < r1; i++) {
            threads[i] = new RowMultiplier(i, A, B, result);
            threads[i].start();
        }

        // Join all threads
        for (int i = 0; i < r1; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return result;
    }

    // -------------------- MAIN PROGRAM --------------------
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter rows of Matrix A: ");
        int r1 = sc.nextInt();

        System.out.print("Enter columns of Matrix A (and rows of Matrix B): ");
        int c1 = sc.nextInt();

        System.out.print("Enter columns of Matrix B: ");
        int c2 = sc.nextInt();

        int[][] A = new int[r1][c1];
        int[][] B = new int[c1][c2];

        System.out.println("\nEnter Matrix A:");
        for (int i = 0; i < r1; i++)
            for (int j = 0; j < c1; j++)
                A[i][j] = sc.nextInt();

        System.out.println("\nEnter Matrix B:");
        for (int i = 0; i < c1; i++)
            for (int j = 0; j < c2; j++)
                B[i][j] = sc.nextInt();

        // -------------------- NORMAL MULTIPLICATION --------------------
        long start1 = System.nanoTime();
        int[][] normalResult = normalMultiply(A, B);
        long end1 = System.nanoTime();

        long normalTime = end1 - start1;

        // -------------------- MULTITHREADED MULTIPLICATION --------------------
        long start2 = System.nanoTime();
        int[][] threadedResult = threadedMultiply(A, B);
        long end2 = System.nanoTime();

        long threadedTime = end2 - start2;

        // -------------------- OUTPUT --------------------
        System.out.println("\nNormal Multiplication Result:");
        printMatrix(normalResult);

        System.out.println("Time (Normal): " + normalTime + " ns");

        System.out.println("\nMultithreaded Multiplication Result:");
        printMatrix(threadedResult);

        System.out.println("Time (Multithreaded): " + threadedTime + " ns");
    }

    public static void printMatrix(int[][] M) {
        for (int[] row : M) {
            for (int val : row)
                System.out.print(val + " ");
            System.out.println();
        }
    }
}
