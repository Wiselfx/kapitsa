#!/usr/bin/perl
# deadline.pl SECONDS command...: runs the command in a process group of its own; past SECONDS the whole group gets
# TERM (servers stop, serve_run.sh cleans up), 15 s later KILL. Exits with the command's status (124 on the deadline).
use strict; use warnings; use POSIX ":sys_wait_h";
my $secs = shift @ARGV;
my $pid = fork();
die "fork: $!" unless defined $pid;
if ($pid == 0) { setpgrp(0, 0); exec @ARGV or die "exec: $!"; }
my $deadline = time() + $secs; my $termed = 0;
while (1) {
  my $r = waitpid($pid, WNOHANG);
  if ($r == $pid) { exit($termed ? 124 : ($? >> 8)); }
  if (!$termed && time() >= $deadline) { print STDERR "deadline: TERM to group $pid\n"; kill 'TERM', -$pid; $termed = time(); }
  if ($termed && time() >= $termed + 15) { print STDERR "deadline: KILL to group $pid\n"; kill 'KILL', -$pid; }
  select(undef, undef, undef, 0.5);
}
