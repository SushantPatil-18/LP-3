import java.util.*;

public class QuickSortAnalysis {

    // ------------------ Deterministic Quick Sort (Pivot = Last Element)
    // ------------------
    public static void deterministicQuickSort(int arr[], int low, int high) {
        if (low < high) {
            int p = deterministicPartition(arr, low, high);
            deterministicQuickSort(arr, low, p - 1);
            deterministicQuickSort(arr, p + 1, high);
        }
    }

    public static int deterministicPartition(int arr[], int low, int high) {
        int pivot = arr[high]; // last element pivot
        int i = low - 1;

        for (int j = low; j < high; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[high];
        arr[high] = temp;
        return i + 1;
    }

    // ------------------ Randomized Quick Sort ------------------
    public static void randomizedQuickSort(int arr[], int low, int high) {
        if (low < high) {

            int p = randomizedPartition(arr, low, high);
            randomizedQuickSort(arr, low, p - 1);
            randomizedQuickSort(arr, p + 1, high);
        }
    }

    public static int randomizedPartition(int arr[], int low, int high) {
        Random rand = new Random();
        int randomIndex = low + rand.nextInt(high - low + 1);

        // Swap random element with last element
        int temp = arr[randomIndex];
        arr[randomIndex] = arr[high];
        arr[high] = temp;

        return deterministicPartition(arr, low, high);
    }

    // ------------------ Main: Analysis ------------------
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter size of array: ");
        int n = sc.nextInt();

        int arr[] = new int[n];
        System.out.println("Enter array elements:");
        for (int i = 0; i < n; i++)
            arr[i] = sc.nextInt();

        int arr1[] = arr.clone();
        int arr2[] = arr.clone();

        // ------------------ Deterministic ------------------
        long start1 = System.nanoTime();
        deterministicQuickSort(arr1, 0, n - 1);
        long end1 = System.nanoTime();
        long timeDet = end1 - start1;

        // ------------------ Randomized ------------------
        long start2 = System.nanoTime();
        randomizedQuickSort(arr2, 0, n - 1);
        long end2 = System.nanoTime();
        long timeRand = end2 - start2;

        // ------------------ Output ------------------
        System.out.println("\nSorted Array (Deterministic): " + Arrays.toString(arr1));
        System.out.println("Time Taken (Deterministic): " + timeDet + " ns");

        System.out.println("\nSorted Array (Randomized): " + Arrays.toString(arr2));
        System.out.println("Time Taken (Randomized): " + timeRand + " ns");
    }
}
