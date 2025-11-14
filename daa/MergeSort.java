import java.util.*;

class MergeSort {

    // ---------- NORMAL MERGE SORT ----------
    public static void mergeSort(int[] arr, int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;

            mergeSort(arr, left, mid);
            mergeSort(arr, mid + 1, right);
            merge(arr, left, mid, right);
        }
    }

    // ---------- MULTITHREADED MERGE SORT ----------
    static class MergeSortThread extends Thread {
        int[] arr;
        int left, right;

        MergeSortThread(int[] arr, int left, int right) {
            this.arr = arr;
            this.left = left;
            this.right = right;
        }

        @Override
        public void run() {
            mergeSortThreaded(arr, left, right);
        }
    }

    public static void mergeSortThreaded(int[] arr, int left, int right) {
        if (left < right) {
            int mid = (left + right) / 2;

            // Create 2 threads
            MergeSortThread t1 = new MergeSortThread(arr, left, mid);
            MergeSortThread t2 = new MergeSortThread(arr, mid + 1, right);

            t1.start();
            t2.start();

            try {
                t1.join();
                t2.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            merge(arr, left, mid, right);
        }
    }

    // ---------- MERGE FUNCTION ----------
    public static void merge(int[] arr, int left, int mid, int right) {
        int n1 = mid - left + 1;
        int n2 = right - mid;

        int[] L = new int[n1];
        int[] R = new int[n2];

        for (int i = 0; i < n1; i++)
            L[i] = arr[left + i];
        for (int j = 0; j < n2; j++)
            R[j] = arr[mid + 1 + j];

        int i = 0, j = 0, k = left;

        while (i < n1 && j < n2) {
            if (L[i] <= R[j])
                arr[k++] = L[i++];
            else
                arr[k++] = R[j++];
        }

        while (i < n1)
            arr[k++] = L[i++];
        while (j < n2)
            arr[k++] = R[j++];
    }

    // ---------- MAIN METHOD ----------
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);
        System.out.print("Enter size of array: ");
        int n = sc.nextInt();

        int[] arr1 = new int[n];
        int[] arr2 = new int[n];

        System.out.println("Enter array elements:");
        for (int i = 0; i < n; i++) {
            arr1[i] = sc.nextInt();
            arr2[i] = arr1[i];
        }

        // ----- NORMAL MERGE SORT -----
        long startNormal = System.nanoTime();
        mergeSort(arr1, 0, n - 1);
        long endNormal = System.nanoTime();

        // ----- MULTITHREADED MERGE SORT -----
        long startThreaded = System.nanoTime();
        mergeSortThreaded(arr2, 0, n - 1);
        long endThreaded = System.nanoTime();

        System.out.println("\nSorted Array using Merge Sort:");
        System.out.println(Arrays.toString(arr1));

        System.out.println("\nSorted Array using Multithreaded Merge Sort:");
        System.out.println(Arrays.toString(arr2));

        System.out.println("\nTime (normal merge sort): " + (endNormal - startNormal) + " ns");
        System.out.println("Time (multithreaded merge sort): " + (endThreaded - startThreaded) + " ns");
    }
}
