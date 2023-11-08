package main

import (
    "os"
    "fmt"
    "github.com/MagnusChase03/GoNN/cli"
)

func main() {
    args := os.Args[1:]
    if (len(args) < 2) {
        PrintUsage()
        return
    }

    err := cli.Execute(args)
    if (err != nil) {
        fmt.Printf("\033[1;31m[ERROR]\033[0m")
        fmt.Println(err)
    }
}

func PrintUsage() {
    fmt.Println("\033[1;31mUsage:\033[0m <command> [<args>]")
}
