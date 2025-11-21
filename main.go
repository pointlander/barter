// Copyright 2025 The Barter Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"

	"github.com/alixaxel/pagerank"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

func main() {
	iris := Load()

	entropy := func(input []Fisher) float64 {
		graph := pagerank.NewGraph()
		for i := range input {
			for ii := range input {
				sum := 0.0
				for iii, value := range input[i].Measures {
					sum += value * input[ii].Measures[iii]
				}
				graph.Link(uint32(i), uint32(ii), sum)
			}
		}
		nodes := make([]float64, len(input))
		entropy := 0.0
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			nodes[node] = rank
		})
		for _, rank := range nodes {
			if rank > 0 {
				entropy += rank * math.Log2(rank)
			}
		}
		return -entropy
	}
	fmt.Println(entropy(iris))
	fmt.Println(entropy(iris[:50]))
	fmt.Println(entropy(iris[50:100]))
	fmt.Println(entropy(iris[100:]))

	rng := rand.New(rand.NewSource(1))
	perm := rng.Perm(len(iris))
	neurons := make([][]Fisher, 3)
	for i := range neurons {
		neurons[i] = make([]Fisher, 50)
	}
	for i := range neurons[0] {
		neurons[0][i] = iris[perm[i]]
	}
	for i := range neurons[1] {
		neurons[1][i] = iris[perm[i+50]]
	}
	for i := range neurons[2] {
		neurons[2][i] = iris[perm[i+100]]
	}
	fmt.Println(entropy(neurons[0]))
	fmt.Println(entropy(neurons[1]))
	fmt.Println(entropy(neurons[2]))

	for range 8 * 1024 {
		a, b := rng.Intn(3), rng.Intn(3)
		x, y := rng.Intn(len(neurons[a])), rng.Intn(len(neurons[b]))
		entropyA, entropyB := entropy(neurons[a]), entropy(neurons[b])
		neurons[a][x], neurons[b][y] = neurons[b][y], neurons[a][x]
		entropyAGain, entropyBGain := entropy(neurons[a]), entropy(neurons[b])
		if !(entropyAGain > entropyA && entropyBGain > entropyB) {
			neurons[b][y], neurons[a][x] = neurons[a][x], neurons[b][y]
		}
	}

	indexes := make(map[int]bool)
	for _, n := range neurons {
		sort.Slice(n, func(i, j int) bool {
			return n[i].Index < n[j].Index
		})
		for _, entry := range n {
			if indexes[entry.Index] {
				panic("dup")
			}
			indexes[entry.Index] = true
			fmt.Println(entry.Index, entry.Label)
		}
		fmt.Println()
	}

	acc := make(map[string][3]int)
	for cluster := range neurons {
		for i := range neurons[cluster] {
			counts := acc[neurons[cluster][i].Label]
			counts[cluster]++
			acc[neurons[cluster][i].Label] = counts
		}
	}
	for i, v := range acc {
		fmt.Println(i, v)
	}
}
